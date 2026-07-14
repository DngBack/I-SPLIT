"""Fit I-SPLIT intervention-covariance subspaces (plus PCA/random baselines)
for every (encoder, layer, factor), from the cached features + pair tables
that scripts/extract_features.py produced. Saves one .npz per (encoder,
layer, factor, method) under results/subspaces/.

Usage: uv run python scripts/fit_subspaces.py --scale pilot
"""

from pathlib import Path

import click
import numpy as np
import pandas as pd

from isplit.config.loader import load_config
from isplit.subspace.baselines import plain_pca, random_subspace_baseline
from isplit.subspace.pipeline import fit_factor_subspace, load_pooled_layer
from isplit.utils.logging import get_logger
from isplit.utils.seeding import set_all_seeds

logger = get_logger(__name__)


def _save_subspace(path: Path, basis: np.ndarray, eigvals: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if eigvals is None:
        np.savez(path, basis=basis)
    else:
        np.savez(path, basis=basis, eigvals=eigvals)


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

    summary_rows = []
    for encoder_cfg in cfg.encoders.active():
        for layer in range(encoder_cfg.num_layers + 1):  # +1 for the input-embedding layer (index 0)
            # unlabeled PCA/random baselines only need a pool of train-split features,
            # not paired data -- reuse the speaker-pair "a" utterances as that pool.
            speaker_train = pair_sets["speaker"][pair_sets["speaker"]["split"] == "train"]
            if speaker_train.empty:
                continue
            pooled_pool = np.stack(
                [
                    load_pooled_layer(features_dir, encoder_cfg.name, "train", uid, None, layer)
                    for uid in speaker_train["a_utt_id"].unique()
                ]
            )

            for factor in cfg.subspace.factors:
                pairs = pair_sets[factor]
                if pairs.empty or (pairs["split"] == "train").sum() < 2:
                    logger.warning("skipping %s/%s/layer%d: not enough train pairs", encoder_cfg.name, factor, layer)
                    continue

                basis, eigvals_used, eigvals_full = fit_factor_subspace(
                    features_dir, encoder_cfg.name, layer, pairs, factor,
                    rank=cfg.subspace.rank, energy_threshold=cfg.subspace.energy_threshold,
                )
                rank = basis.shape[1]
                _save_subspace(
                    subspaces_dir / encoder_cfg.name / factor / f"layer_{layer:02d}__isplit.npz", basis, eigvals_full
                )

                pca_basis = plain_pca(pooled_pool, rank=rank)
                _save_subspace(subspaces_dir / encoder_cfg.name / factor / f"layer_{layer:02d}__pca.npz", pca_basis, None)

                random_basis = random_subspace_baseline(basis.shape[0], rank, seed=cfg.seed)
                _save_subspace(
                    subspaces_dir / encoder_cfg.name / factor / f"layer_{layer:02d}__random.npz", random_basis, None
                )

                energy_fraction = float(np.clip(eigvals_used, 0, None).sum() / max(np.clip(eigvals_full, 0, None).sum(), 1e-12))
                summary_rows.append(
                    {
                        "encoder": encoder_cfg.name,
                        "layer": layer,
                        "factor": factor,
                        "rank": rank,
                        "top_rank_energy_fraction": energy_fraction,
                    }
                )
            logger.info("%s layer %d: done", encoder_cfg.name, layer)

    summary = pd.DataFrame(summary_rows)
    summary_path = Path(cfg.results_dir) / "tables" / "subspace_fit_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote subspace fit summary to %s", summary_path)


if __name__ == "__main__":
    main()
