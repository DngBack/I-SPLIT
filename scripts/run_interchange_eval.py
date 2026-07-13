"""Train per-factor probes and run the interchange-intervention causal
evaluation (Preserve / Transfer / CSS) for every (encoder, layer, factor),
using the I-SPLIT subspaces from scripts/fit_subspaces.py. Produces the main
layer-wise audit table (decodability, IEI, CSS side by side) that
eval.probe_vs_intervention consumes for Claim 1.

Usage: uv run python scripts/run_interchange_eval.py --scale pilot
"""

from pathlib import Path

import click
import numpy as np
import pandas as pd

from isplit.causal.entanglement import irreducible_entanglement_index
from isplit.config.loader import load_config
from isplit.eval.layerwise_audit import layerwise_causal_selectivity, layerwise_decodability
from isplit.probes.linear import LinearProbe
from isplit.subspace.pipeline import load_pooled_layer
from isplit.utils.logging import get_logger
from isplit.utils.seeding import set_all_seeds

logger = get_logger(__name__)

LABEL_COL = {"speaker": "speaker_id", "content": "prompt_id"}


def _load_subspace(subspaces_dir: Path, encoder: str, factor: str, layer: int, method: str = "isplit") -> np.ndarray:
    path = subspaces_dir / encoder / factor / f"layer_{layer:02d}__{method}.npz"
    with np.load(path) as data:
        return data["basis"]


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
        factor: pd.read_parquet(manifests_dir / f"pairs_{factor}.parquet") for factor in cfg.subspace.factors
    }
    train_speakers = set(pair_sets["speaker"].loc[pair_sets["speaker"]["split"] == "train", "a_speaker_id"]) | set(
        pair_sets["speaker"].loc[pair_sets["speaker"]["split"] == "train", "b_speaker_id"]
    )
    utt_to_split = {
        uid: ("train" if sid in train_speakers else "held_out")
        for uid, sid in zip(vctk["utt_id"], vctk["speaker_id"], strict=True)
    }

    audit_rows = []
    for encoder_cfg in cfg.encoders.encoders:
        for layer in range(encoder_cfg.num_layers + 1):
            factor_subspaces = {}
            for factor in cfg.subspace.factors:
                path = subspaces_dir / encoder_cfg.name / factor / f"layer_{layer:02d}__isplit.npz"
                if not path.exists():
                    continue
                factor_subspaces[factor] = _load_subspace(subspaces_dir, encoder_cfg.name, factor, layer)

            # Train one probe per label column up front so each factor's causal
            # eval can use the *other* factor's probe as its off-target Preserve
            # check (e.g. swapping speaker should preserve content, and vice versa).
            train_rows = vctk[vctk["utt_id"].map(utt_to_split) == "train"]
            label_probes: dict[str, LinearProbe] = {}
            if not train_rows.empty:
                x_train = np.stack(
                    [
                        load_pooled_layer(features_dir, encoder_cfg.name, "train", uid, None, layer)
                        for uid in train_rows["utt_id"]
                    ]
                )
                for label_factor, label_col in LABEL_COL.items():
                    if label_col in vctk.columns:
                        y_train = train_rows[label_col].to_numpy()
                        label_probes[label_factor] = LinearProbe().fit(x_train, y_train)

            for factor in ("speaker", "content"):
                if factor not in factor_subspaces or factor not in label_probes:
                    continue
                decodability = layerwise_decodability(
                    features_dir, encoder_cfg.name, layer, vctk, LABEL_COL[factor], utt_to_split
                )

                iei_values = [
                    irreducible_entanglement_index(factor_subspaces[factor], factor_subspaces[other])
                    for other in factor_subspaces
                    if other != factor
                ]
                iei = float(np.mean(iei_values)) if iei_values else float("nan")

                held_out_pairs = pair_sets[factor][pair_sets[factor]["split"] == "held_out"]
                css = float("nan")
                if not held_out_pairs.empty:
                    target_probe = label_probes[factor]
                    # off-target probe: any other label with a trained probe (e.g.
                    # content probe when swapping speaker) proxies Preserve
                    off_target = next((p for f, p in label_probes.items() if f != factor), None)
                    css = layerwise_causal_selectivity(
                        features_dir, encoder_cfg.name, layer, factor_subspaces,
                        held_out_pairs, target_probe, content_probe=off_target, factor=factor,
                    )

                audit_rows.append(
                    {
                        "encoder": encoder_cfg.name,
                        "layer": layer,
                        "factor": factor,
                        "decodability": decodability,
                        "iei": iei,
                        "css": css,
                    }
                )
            logger.info("%s layer %d: audit computed", encoder_cfg.name, layer)

    audit_df = pd.DataFrame(audit_rows)
    out_path = Path(cfg.results_dir) / "tables" / "layerwise_audit.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(out_path, index=False)
    logger.info("Wrote layer-wise audit table to %s (%d rows)", out_path, len(audit_df))


if __name__ == "__main__":
    main()
