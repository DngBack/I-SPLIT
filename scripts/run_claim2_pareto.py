"""Claim 2 on the real encoder features: at matched nuisance erasure, does the
regularized *oblique* projection preserve content better than *orthogonal*
removal of the nuisance subspace?

scripts/run_theory_validation.py already checks this on synthetic data, where the
principal angle between the two subspaces is a knob we control. This script runs
the same comparison on the cached frozen-encoder features with the I-SPLIT
subspaces from scripts/fit_subspaces.py: for every (encoder, layer) it stacks
[content_basis | nuisance_basis], sweeps the ridge strength tau, and compares the
resulting (nuisance-erasure, content-preservation) curve against the single
operating point of orthogonal nuisance removal.

Two conventions are worth stating, because both are easy to get wrong:

* Preservation and erasure are scored against the labels the probes read off the
  *original, unprojected* representation -- not the manifest ground truth. That is
  eval.pareto's contract, and it is the right one: Claim 2 asks what the projection
  does to the information a probe can pull out of a representation, so the
  unprojected representation is the reference. It also keeps both metrics
  well-defined where a probe does not generalize (the speaker probe, in
  particular, has never seen a held-out speaker's voice).
* Everything is fitted on train speakers (probes, subspaces) and evaluated on
  held-out speakers.

Eval set per nuisance factor
----------------------------
speaker
    Held-out speakers' clean utterances; the nuisance probe predicts speaker id.
environment / channel
    Held-out speakers' *augmented* endpoints (both conditions of every pair in
    pairs_<factor>.parquet); the nuisance probe predicts the condition label
    (SNR level / channel type) -- the same probe target run_interchange_eval.py
    uses for these two factors.

    These two deliberately do not reuse the clean eval set. On clean features
    every utterance carries the same condition label, so a nuisance probe there
    would have exactly one class: its accuracy is 1.0 by construction, and
    "erasure" would only measure how far a projection pushes a representation off
    the clean manifold -- there is no nuisance information in that eval set to
    erase. Rather than invent a label, environment/channel are evaluated where
    their nuisance actually varies. The content probe stays the same one
    throughout (prompt id, fitted on clean train features).

Usage: uv run python scripts/run_claim2_pareto.py --scale pilot
"""

from pathlib import Path

import click
import numpy as np
import pandas as pd

from isplit.config.loader import load_config
from isplit.data.augment import encode_channel_condition, encode_environment_condition
from isplit.encoders.cache import cache_path
from isplit.eval.pareto import pareto_frontier_area, preservation_erasure_curve
from isplit.probes.linear import LinearProbe
from isplit.subspace.pipeline import load_pooled_layer
from isplit.utils.logging import get_logger
from isplit.utils.seeding import set_all_seeds

logger = get_logger(__name__)

CONTENT_FACTOR = "content"
CONTENT_LABEL = "prompt_id"
NUISANCE_FACTORS = ("speaker", "environment", "channel")
CONDITION_FACTORS = ("environment", "channel")

# The feature cache is written per (split, utterance, condition); a partially
# extracted cache (e.g. an extraction still running, or a manifest regenerated at
# a larger scale) would otherwise blow up mid-probe. Rows without a cached file
# are dropped, and an (encoder, layer, factor) cell with too few surviving eval
# utterances is skipped rather than reported as a noisy number.
MIN_EVAL_UTTS = 20


def _load_isplit_basis(subspaces_dir: Path, encoder: str, factor: str, layer: int) -> np.ndarray | None:
    path = subspaces_dir / encoder / factor / f"layer_{layer:02d}__isplit.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return data["basis"]


def _cached_only(features_dir: Path, encoder: str, rows: pd.DataFrame) -> pd.DataFrame:
    keep = [
        cache_path(features_dir, encoder, r["split"], r["utt_id"], r["condition"]).exists()
        for _, r in rows.iterrows()
    ]
    return rows[np.asarray(keep, dtype=bool)] if len(rows) else rows


def _pooled(features_dir: Path, encoder: str, layer: int, rows: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [
            load_pooled_layer(features_dir, encoder, r["split"], r["utt_id"], r["condition"], layer)
            for _, r in rows.iterrows()
        ]
    )


def _environment_label(noise_id, snr_db) -> str:
    """Which noise level a representation was produced under -- the thing an
    environment intervention is supposed to move. Matches run_interchange_eval.py.
    """
    if noise_id is None or (isinstance(noise_id, float) and np.isnan(noise_id)) or noise_id == "clean":
        return "clean"
    return f"snr={float(snr_db):g}"


def condition_endpoints(pairs: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Both endpoints of every environment/channel pair, as labelled (utt_id,
    condition, split) rows -- the examples the condition probe trains/evals on.
    """
    rows = []
    for _, row in pairs.iterrows():
        for side in ("a", "b"):
            if factor == "environment":
                noise_id, snr = row[f"{side}_noise_id"], row[f"{side}_snr_db"]
                condition = encode_environment_condition(noise_id, snr)
                label = _environment_label(noise_id, snr)
            else:
                channel = row[f"{side}_channel"]
                condition = encode_channel_condition(channel)
                label = channel
            rows.append(
                {
                    "utt_id": row["base_utt_id"],
                    "condition": condition,
                    "split": row["split"],
                    "label": label,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["utt_id", "condition", "split", "label"])
    return pd.DataFrame(rows).drop_duplicates(subset=["utt_id", "condition"]).reset_index(drop=True)


def clean_rows(vctk: pd.DataFrame, utt_to_split: dict[str, str], split: str) -> pd.DataFrame:
    """Clean (unaugmented) utterances of one speaker split, in the (utt_id, split,
    condition, label) shape the loaders above expect.
    """
    rows = vctk[vctk["utt_id"].map(utt_to_split) == split]
    return pd.DataFrame(
        {
            "utt_id": rows["utt_id"].to_numpy(),
            "condition": None,
            "split": split,
            "label": rows["speaker_id"].to_numpy(),
            CONTENT_LABEL: rows[CONTENT_LABEL].to_numpy(),
        }
    )


def curve_metrics(curve: pd.DataFrame) -> dict[str, float]:
    """Reduce one (encoder, layer, factor) tau-sweep to the two comparisons Claim 2
    actually makes.

    Frontier area: the oblique curve has a tau knob and therefore a frontier;
    orthogonal removal has none, so its comparable "frontier" is the constant curve
    at its own preservation, taken over the same erasure support. The area
    comparison then reads as: across the erasure range oblique can reach, does it
    hold more content than orthogonal's single operating point does?

    Matched erasure: the literal statement of the claim -- at orthogonal's erasure
    level, interpolate the oblique curve's preservation and compare. np.interp
    clamps outside the sweep's range, so `matched_in_range` records whether the
    comparison is an interpolation (trustworthy) or a clamp (not).
    """
    oblique = curve[curve["method"] == "oblique"]
    orth = curve[curve["method"] == "orthogonal"].iloc[0]
    orth_erasure = float(orth["nuisance_erasure"])
    orth_preserve = float(orth["content_preservation"])

    x_min = float(oblique["nuisance_erasure"].min())
    x_max = float(oblique["nuisance_erasure"].max())
    oblique_area = pareto_frontier_area(oblique)
    orth_area = pareto_frontier_area(
        pd.DataFrame(
            {"nuisance_erasure": [x_min, x_max], "content_preservation": [orth_preserve, orth_preserve]}
        )
    )

    sweep = oblique.sort_values("nuisance_erasure")
    matched = float(
        np.interp(orth_erasure, sweep["nuisance_erasure"].to_numpy(), sweep["content_preservation"].to_numpy())
    )
    return {
        "oblique_area": oblique_area,
        "orthogonal_area": orth_area,
        "area_delta": oblique_area - orth_area,
        "orthogonal_erasure": orth_erasure,
        "orthogonal_preservation": orth_preserve,
        "oblique_preservation_at_matched_erasure": matched,
        "matched_delta": matched - orth_preserve,
        "matched_in_range": bool(x_min <= orth_erasure <= x_max),
    }


def summarize(layer_df: pd.DataFrame) -> pd.DataFrame:
    """Per (encoder, nuisance factor): the two comparisons averaged over layers,
    plus how often oblique wins layer-by-layer (a mean can be carried by one layer).
    """
    rows = []
    for (encoder, factor), group in layer_df.groupby(["encoder", "nuisance_factor"]):
        rows.append(
            {
                "encoder": encoder,
                "nuisance_factor": factor,
                "n_layers": int(len(group)),
                "oblique_area_mean": float(group["oblique_area"].mean()),
                "orthogonal_area_mean": float(group["orthogonal_area"].mean()),
                "area_delta_mean": float(group["area_delta"].mean()),
                "frac_layers_oblique_wins_area": float((group["area_delta"] > 0).mean()),
                "orthogonal_erasure_mean": float(group["orthogonal_erasure"].mean()),
                "orthogonal_preservation_mean": float(group["orthogonal_preservation"].mean()),
                "oblique_preservation_at_matched_erasure_mean": float(
                    group["oblique_preservation_at_matched_erasure"].mean()
                ),
                "matched_delta_mean": float(group["matched_delta"].mean()),
                "frac_layers_oblique_wins_matched": float((group["matched_delta"] > 0).mean()),
                "frac_layers_matched_in_range": float(group["matched_in_range"].mean()),
                "oblique_wins": bool(group["area_delta"].mean() > 0),
            }
        )
    return pd.DataFrame(rows)


def _plot_pareto(pareto_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    factors = sorted(pareto_df["nuisance_factor"].unique())
    encoders = sorted(pareto_df["encoder"].unique())
    colors = dict(zip(encoders, plt.get_cmap("tab10").colors, strict=False))

    fig, axes = plt.subplots(1, len(factors), figsize=(5.5 * len(factors), 4.5), squeeze=False)
    for ax, factor in zip(axes[0], factors, strict=False):
        group = pareto_df[pareto_df["nuisance_factor"] == factor]
        for encoder in encoders:
            enc = group[group["encoder"] == encoder]
            if enc.empty:
                continue
            color = colors[encoder]
            oblique = enc[enc["method"] == "oblique"]
            orth = enc[enc["method"] == "orthogonal"]
            # every (encoder, layer) cell, faint, then that encoder's mean curve on top
            ax.scatter(
                oblique["nuisance_erasure"], oblique["content_preservation"],
                s=10, alpha=0.25, color=color,
            )
            ax.scatter(
                orth["nuisance_erasure"], orth["content_preservation"],
                s=25, alpha=0.35, color=color, marker="x",
            )
            mean_curve = oblique.groupby("tau")[["nuisance_erasure", "content_preservation"]].mean()
            mean_curve = mean_curve.sort_values("nuisance_erasure")
            ax.plot(
                mean_curve["nuisance_erasure"], mean_curve["content_preservation"],
                marker="o", color=color, label=f"{encoder} oblique (tau sweep)",
            )
            ax.scatter(
                [orth["nuisance_erasure"].mean()], [orth["content_preservation"].mean()],
                s=170, color=color, marker="*", edgecolor="black", linewidth=0.5, zorder=5,
                label=f"{encoder} orthogonal",
            )
        ax.set_title(f"nuisance={factor}")
        ax.set_xlabel("nuisance erasure (1 - probe acc. vs. original)")
        ax.set_ylabel("content preservation (probe acc. vs. original)")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@click.command()
@click.option("--scale", default="pilot", type=click.Choice(["pilot", "full"]))
@click.option("--config-dir", default="configs")
def main(scale: str, config_dir: str) -> None:
    cfg = load_config(scale=scale, config_dir=config_dir)
    set_all_seeds(cfg.seed)

    manifests_dir = Path(cfg.results_dir) / "manifests"
    features_dir = Path(cfg.results_dir) / "features"
    subspaces_dir = Path(cfg.results_dir) / "subspaces"

    vctk = pd.read_parquet(manifests_dir / "vctk.parquet")
    pair_sets = {
        factor: pd.read_parquet(manifests_dir / f"pairs_{factor}.parquet")
        for factor in cfg.subspace.factors
    }
    speaker_pairs = pair_sets["speaker"]
    train_speakers = set(speaker_pairs.loc[speaker_pairs["split"] == "train", "a_speaker_id"]) | set(
        speaker_pairs.loc[speaker_pairs["split"] == "train", "b_speaker_id"]
    )
    utt_to_split = {
        uid: ("train" if sid in train_speakers else "held_out")
        for uid, sid in zip(vctk["utt_id"], vctk["speaker_id"], strict=True)
    }
    endpoints = {
        factor: condition_endpoints(pair_sets[factor], factor)
        for factor in CONDITION_FACTORS
        if factor in pair_sets
    }
    utt_to_prompt = dict(zip(vctk["utt_id"], vctk[CONTENT_LABEL], strict=True))

    train_clean = clean_rows(vctk, utt_to_split, "train")
    eval_clean = clean_rows(vctk, utt_to_split, "held_out")

    pareto_rows, layer_rows = [], []
    for encoder_cfg in cfg.encoders.active():
        encoder = encoder_cfg.name
        enc_train_clean = _cached_only(features_dir, encoder, train_clean)
        enc_eval_clean = _cached_only(features_dir, encoder, eval_clean)
        if len(enc_train_clean) < len(train_clean) or len(enc_eval_clean) < len(eval_clean):
            logger.warning(
                "%s: %d/%d train and %d/%d held-out clean utterances are cached; using what is there",
                encoder, len(enc_train_clean), len(train_clean), len(enc_eval_clean), len(eval_clean),
            )
        if enc_train_clean.empty or len(enc_eval_clean) < MIN_EVAL_UTTS:
            logger.warning("%s: not enough cached clean features, skipping encoder", encoder)
            continue

        enc_endpoints = {
            factor: {
                split: _cached_only(features_dir, encoder, df[df["split"] == split])
                for split in ("train", "held_out")
            }
            for factor, df in endpoints.items()
        }

        for layer in range(encoder_cfg.num_layers + 1):
            content_basis = _load_isplit_basis(subspaces_dir, encoder, CONTENT_FACTOR, layer)
            if content_basis is None:
                logger.warning("%s layer %d: no content subspace, skipping layer", encoder, layer)
                continue

            x_train_clean = _pooled(features_dir, encoder, layer, enc_train_clean)
            content_probe = LinearProbe().fit(x_train_clean, enc_train_clean[CONTENT_LABEL].to_numpy())
            speaker_probe = LinearProbe().fit(x_train_clean, enc_train_clean["label"].to_numpy())

            for factor in NUISANCE_FACTORS:
                nuisance_basis = _load_isplit_basis(subspaces_dir, encoder, factor, layer)
                if nuisance_basis is None:
                    logger.warning("%s layer %d: no %s subspace, skipping", encoder, layer, factor)
                    continue

                if factor == "speaker":
                    eval_rows = enc_eval_clean
                    nuisance_probe = speaker_probe
                else:
                    if factor not in enc_endpoints:
                        continue
                    train_ep = enc_endpoints[factor]["train"]
                    eval_rows = enc_endpoints[factor]["held_out"]
                    if train_ep.empty or train_ep["label"].nunique() < 2:
                        logger.warning("%s layer %d: no usable %s probe, skipping", encoder, layer, factor)
                        continue
                    nuisance_probe = LinearProbe().fit(
                        _pooled(features_dir, encoder, layer, train_ep), train_ep["label"].to_numpy()
                    )

                if len(eval_rows) < MIN_EVAL_UTTS:
                    logger.warning(
                        "%s layer %d: only %d cached %s eval examples (<%d), skipping",
                        encoder, layer, len(eval_rows), factor, MIN_EVAL_UTTS,
                    )
                    continue

                x_eval = _pooled(features_dir, encoder, layer, eval_rows)
                # eval.pareto's contract: the reference labels are what each probe reads
                # off the *original* representation, not the manifest ground truth.
                content_labels_true = content_probe.predict(x_eval)
                nuisance_labels_true = nuisance_probe.predict(x_eval)

                curve = preservation_erasure_curve(
                    content_features=x_eval,
                    nuisance_features_true=x_eval,
                    content_basis=content_basis,
                    nuisance_basis=nuisance_basis,
                    tau_values=list(cfg.subspace.tau_values),
                    content_probe=content_probe,
                    nuisance_probe=nuisance_probe,
                    content_labels_true=content_labels_true,
                    nuisance_labels_true=nuisance_labels_true,
                )

                # Sanity diagnostics: how well each probe does against the *true* labels
                # on this eval set. A probe near chance makes its curve uninterpretable.
                true_content = np.asarray([utt_to_prompt[u] for u in eval_rows["utt_id"]])
                content_acc = content_probe.score(x_eval, true_content)
                nuisance_acc = nuisance_probe.score(x_eval, eval_rows["label"].to_numpy())

                curve = curve.assign(
                    encoder=encoder,
                    layer=layer,
                    nuisance_factor=factor,
                    content_rank=int(content_basis.shape[1]),
                    nuisance_rank=int(nuisance_basis.shape[1]),
                    n_eval=int(len(eval_rows)),
                    content_probe_acc_true=content_acc,
                    nuisance_probe_acc_true=nuisance_acc,
                )
                pareto_rows.append(curve)
                layer_rows.append(
                    {
                        "encoder": encoder,
                        "layer": layer,
                        "nuisance_factor": factor,
                        **curve_metrics(curve),
                    }
                )
            logger.info("%s layer %d: pareto curves computed", encoder, layer)

    if not pareto_rows:
        logger.error("no (encoder, layer, factor) cell had both a subspace and cached features")
        return

    columns = [
        "encoder", "layer", "nuisance_factor", "method", "tau",
        "content_preservation", "nuisance_erasure",
        "content_rank", "nuisance_rank", "n_eval",
        "content_probe_acc_true", "nuisance_probe_acc_true",
    ]
    pareto_df = pd.concat(pareto_rows, ignore_index=True)[columns]
    layer_df = pd.DataFrame(layer_rows)
    summary = summarize(layer_df)

    tables_dir = Path(cfg.results_dir) / "tables"
    figures_dir = Path(cfg.results_dir) / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pareto_df.to_csv(tables_dir / "claim2_pareto.csv", index=False)
    summary.to_csv(tables_dir / "claim2_summary.csv", index=False)
    _plot_pareto(pareto_df, figures_dir / "claim2_pareto.png")

    logger.info("Claim 2 -- oblique vs. orthogonal on real features:\n%s", summary.to_string(index=False))
    logger.info(
        "Wrote %s (%d rows), %s and %s",
        tables_dir / "claim2_pareto.csv", len(pareto_df),
        tables_dir / "claim2_summary.csv", figures_dir / "claim2_pareto.png",
    )


if __name__ == "__main__":
    main()
