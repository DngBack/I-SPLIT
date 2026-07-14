"""Train per-factor probes and run the interchange-intervention causal
evaluation (Preserve / Transfer / CSS) for every (encoder, layer, factor,
method), using the subspaces from scripts/fit_subspaces.py. Produces the main
layer-wise audit table (decodability, IEI, CSS side by side) that
eval.probe_vs_intervention consumes for Claim 1.

Three things here are what make Claim 1 testable rather than merely stated:

* **Baselines are evaluated, not just fitted.** CSS is computed for the I-SPLIT
  subspace *and* for the PCA and random-subspace controls at the same rank. A
  CSS number with no control says nothing -- the question is always "better than
  a random subspace of the same size?".
* **All four factors are audited**, not just speaker/content. Environment and
  channel get probes over their condition labels (SNR level / channel type), so
  their subspaces face the same interchange test as the others.
* **Content decodability is measured two ways**: prompt-id classification (which
  saturates at 1.0 -- 24 fixed prompts is too easy, leaving no variance to
  correlate against) and character-error-rate from a CTC probe on frame-level
  features, which does not saturate.

Usage: uv run python scripts/run_interchange_eval.py --scale pilot
"""

from pathlib import Path

import click
import numpy as np
import pandas as pd

from isplit.causal.entanglement import irreducible_entanglement_index
from isplit.config.loader import load_config
from isplit.data.augment import encode_channel_condition, encode_environment_condition
from isplit.eval.layerwise_audit import layerwise_causal_selectivity, layerwise_decodability
from isplit.probes.ctc_head import CTCHead, cer
from isplit.probes.linear import LinearProbe
from isplit.subspace.pipeline import load_frames_layer, load_pooled_layer
from isplit.utils.logging import get_logger
from isplit.utils.seeding import set_all_seeds

logger = get_logger(__name__)

LABEL_COL = {"speaker": "speaker_id", "content": "prompt_id"}
CONDITION_FACTORS = ("environment", "channel")
METHODS = ("isplit", "pca", "random")

# The CTC probe is the one genuinely expensive probe here (frame-level, one fit
# per encoder x layer), so it gets a utterance budget rather than the full split.
CTC_MAX_TRAIN_UTTS = 300
CTC_MAX_EVAL_UTTS = 100
# CTCHead.fit does one full-batch gradient step per "epoch", not a pass over
# mini-batches -- 40 was 40 total optimizer steps, nowhere near enough for a
# from-scratch linear CTC head. Empirically (scratchpad diagnostic on cached
# wav2vec2-base layer-6 features) CER keeps dropping through ~800 steps and is
# flat from there to 3000, so 800 is the plateau, not an arbitrary bump.
CTC_EPOCHS = 800


def _load_subspace(subspaces_dir: Path, encoder: str, factor: str, layer: int, method: str) -> np.ndarray | None:
    path = subspaces_dir / encoder / factor / f"layer_{layer:02d}__{method}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return data["basis"]


def _environment_label(noise_id, snr_db) -> str:
    """Class label for the environment probe: which noise level a representation
    was produced under (clean vs each SNR), i.e. the thing an environment
    intervention is supposed to move.
    """
    if noise_id is None or (isinstance(noise_id, float) and np.isnan(noise_id)) or noise_id == "clean":
        return "clean"
    return f"snr={float(snr_db):g}"


def condition_endpoints(pairs: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Both endpoints of every pair as (utt_id, condition, split, label) rows --
    the labelled examples an environment/channel probe trains and evals on.
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


def _pooled(features_dir: Path, encoder: str, layer: int, rows: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [
            load_pooled_layer(features_dir, encoder, r["split"], r["utt_id"], r["condition"], layer)
            for _, r in rows.iterrows()
        ]
    )


def condition_probe_and_decodability(
    features_dir: Path, encoder: str, layer: int, endpoints: pd.DataFrame
) -> tuple[LinearProbe | None, float]:
    """Probe predicting the condition label (SNR level / channel type) from a
    representation, trained on train-speaker endpoints and scored on held-out ones.
    """
    train_rows = endpoints[endpoints["split"] == "train"]
    held_rows = endpoints[endpoints["split"] == "held_out"]
    if train_rows.empty or train_rows["label"].nunique() < 2:
        return None, float("nan")

    probe = LinearProbe().fit(_pooled(features_dir, encoder, layer, train_rows), train_rows["label"].to_numpy())
    if held_rows.empty or not set(held_rows["label"]) & set(train_rows["label"]):
        return probe, float("nan")
    score = probe.score(_pooled(features_dir, encoder, layer, held_rows), held_rows["label"].to_numpy())
    return probe, float(score)


def content_cer(
    features_dir: Path,
    encoder: str,
    layer: int,
    vctk: pd.DataFrame,
    utt_to_split: dict[str, str],
    seed: int,
    device: str = "cpu",
) -> float:
    """Character error rate of a CTC probe reading text off frame-level features.

    This is the non-saturating content measure. Prompt-id accuracy pins at 1.0
    across nearly every layer (24 classes, trivially separable), and a metric
    with no variance cannot correlate with anything -- so a "decodability does
    not predict CSS" result read off it would be an artifact of the ceiling, not
    a finding.
    """
    train_rows = vctk[vctk["utt_id"].map(utt_to_split) == "train"]
    held_rows = vctk[vctk["utt_id"].map(utt_to_split) == "held_out"]
    if train_rows.empty or held_rows.empty:
        return float("nan")

    rng = np.random.default_rng(seed)
    if len(train_rows) > CTC_MAX_TRAIN_UTTS:
        train_rows = train_rows.iloc[rng.permutation(len(train_rows))[:CTC_MAX_TRAIN_UTTS]]
    if len(held_rows) > CTC_MAX_EVAL_UTTS:
        held_rows = held_rows.iloc[rng.permutation(len(held_rows))[:CTC_MAX_EVAL_UTTS]]

    def _frames(rows: pd.DataFrame) -> list[np.ndarray]:
        return [
            load_frames_layer(features_dir, encoder, utt_to_split[uid], uid, None, layer)
            for uid in rows["utt_id"]
        ]

    train_feats = _frames(train_rows)
    head = CTCHead(feature_dim=train_feats[0].shape[1], epochs=CTC_EPOCHS, seed=seed, device=device)
    head.fit(train_feats, train_rows["text"].tolist())

    errors = [
        cer(head.predict_text(feat), text)
        for feat, text in zip(_frames(held_rows), held_rows["text"].tolist(), strict=True)
    ]
    return float(np.mean(errors))


@click.command()
@click.option("--scale", default="pilot", type=click.Choice(["pilot", "full"]))
@click.option("--config-dir", default="configs")
@click.option("--skip-cer", is_flag=True, help="skip the (slow) CTC content probe")
def main(scale: str, config_dir: str, skip_cer: bool) -> None:
    cfg = load_config(scale=scale, config_dir=config_dir)
    set_all_seeds(cfg.seed)

    manifests_dir = Path(cfg.results_dir) / "manifests"
    features_dir = Path(cfg.results_dir) / "features"
    subspaces_dir = Path(cfg.results_dir) / "subspaces"

    vctk = pd.read_parquet(manifests_dir / "vctk.parquet")
    pair_sets = {
        factor: pd.read_parquet(manifests_dir / f"pairs_{factor}.parquet") for factor in cfg.subspace.factors
    }
    speaker_pairs = pair_sets["speaker"]
    train_speakers = set(speaker_pairs.loc[speaker_pairs["split"] == "train", "a_speaker_id"]) | set(
        speaker_pairs.loc[speaker_pairs["split"] == "train", "b_speaker_id"]
    )
    utt_to_split = {
        uid: ("train" if sid in train_speakers else "held_out")
        for uid, sid in zip(vctk["utt_id"], vctk["speaker_id"], strict=True)
    }
    endpoints = {f: condition_endpoints(pair_sets[f], f) for f in CONDITION_FACTORS if f in pair_sets}

    # Each (encoder, layer) takes minutes; checkpoint after every one so a
    # killed/interrupted run resumes instead of recomputing already-audited
    # layers from scratch.
    out_path = Path(cfg.results_dir) / "tables" / "layerwise_audit.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    done_layers = set()
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        audit_rows = existing_df.to_dict("records")
        done_layers = {(enc, int(l)) for enc, l in zip(existing_df["encoder"], existing_df["layer"], strict=True)}
        logger.info("Resuming from %s: %d (encoder, layer) combos already computed", out_path, len(done_layers))

    for encoder_cfg in cfg.encoders.active():
        for layer in range(encoder_cfg.num_layers + 1):
            if (encoder_cfg.name, layer) in done_layers:
                continue
            subspaces = {
                method: {
                    factor: basis
                    for factor in cfg.subspace.factors
                    if (basis := _load_subspace(subspaces_dir, encoder_cfg.name, factor, layer, method)) is not None
                }
                for method in METHODS
            }

            # Utterance-level probes over clean features (speaker id, prompt id).
            train_rows = vctk[vctk["utt_id"].map(utt_to_split) == "train"]
            probes: dict[str, LinearProbe] = {}
            if not train_rows.empty:
                x_train = np.stack(
                    [
                        load_pooled_layer(features_dir, encoder_cfg.name, "train", uid, None, layer)
                        for uid in train_rows["utt_id"]
                    ]
                )
                for factor, label_col in LABEL_COL.items():
                    if label_col in vctk.columns:
                        probes[factor] = LinearProbe().fit(x_train, train_rows[label_col].to_numpy())

            # Condition-level probes over augmented features (SNR level, channel type).
            decodability: dict[str, float] = {}
            for factor in CONDITION_FACTORS:
                if factor not in endpoints or endpoints[factor].empty:
                    continue
                probe, score = condition_probe_and_decodability(
                    features_dir, encoder_cfg.name, layer, endpoints[factor]
                )
                if probe is not None:
                    probes[factor] = probe
                    decodability[factor] = score

            for factor in LABEL_COL:
                if factor in probes:
                    decodability[factor] = layerwise_decodability(
                        features_dir, encoder_cfg.name, layer, vctk, LABEL_COL[factor], utt_to_split
                    )

            layer_cer = (
                float("nan")
                if skip_cer
                else content_cer(
                    features_dir, encoder_cfg.name, layer, vctk, utt_to_split, cfg.seed, device=cfg.encoders.device
                )
            )

            for factor in cfg.subspace.factors:
                if factor not in probes or factor not in subspaces["isplit"]:
                    continue

                iei_values = [
                    irreducible_entanglement_index(subspaces["isplit"][factor], subspaces["isplit"][other])
                    for other in subspaces["isplit"]
                    if other != factor
                ]
                iei = float(np.mean(iei_values)) if iei_values else float("nan")

                held_out_pairs = pair_sets[factor][pair_sets[factor]["split"] == "held_out"]
                # Preserve is checked with an off-target probe: swapping speaker
                # should leave content where it was, and vice versa.
                off_target = probes.get("content" if factor != "content" else "speaker")

                for method in METHODS:
                    basis = subspaces[method].get(factor)
                    css = float("nan")
                    if basis is not None and not held_out_pairs.empty:
                        css = layerwise_causal_selectivity(
                            features_dir, encoder_cfg.name, layer, {factor: basis},
                            held_out_pairs, probes[factor], content_probe=off_target, factor=factor,
                        )
                    audit_rows.append(
                        {
                            "encoder": encoder_cfg.name,
                            "layer": layer,
                            "factor": factor,
                            "method": method,
                            "decodability": decodability.get(factor, float("nan")),
                            "content_cer": layer_cer if factor == "content" else float("nan"),
                            "iei": iei,
                            "css": css,
                            "rank": int(basis.shape[1]) if basis is not None else 0,
                            "n_held_out_pairs": int(len(held_out_pairs)),
                        }
                    )
            pd.DataFrame(audit_rows).to_csv(out_path, index=False)
            logger.info(
                "%s layer %d: audit computed (content CER=%.3f)", encoder_cfg.name, layer, layer_cer
            )

    logger.info("Wrote layer-wise audit table to %s (%d rows)", out_path, len(audit_rows))


if __name__ == "__main__":
    main()
