"""Assemble the paper's core tables/figures from the outputs of the earlier
scripts: Claim 1 (probe-vs-intervention correlation + ranking disagreement),
Claim 2 (oblique-vs-orthogonal Pareto summary), and a plot of the layer-wise
audit. Run after extract_features.py -> fit_subspaces.py -> run_interchange_eval.py.

Usage: uv run python scripts/make_paper_tables.py --scale pilot
"""

from pathlib import Path

import click
import pandas as pd

from isplit.config.loader import load_config
from isplit.eval.probe_vs_intervention import correlate_metrics, ranking_disagreement
from isplit.utils.logging import get_logger

logger = get_logger(__name__)


@click.command()
@click.option("--scale", default="pilot", type=click.Choice(["pilot", "full"]))
@click.option("--config-dir", default="configs")
def main(scale: str, config_dir: str) -> None:
    cfg = load_config(scale=scale, config_dir=config_dir)
    tables_dir = Path(cfg.results_dir) / "tables"
    figures_dir = Path(cfg.results_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    audit_path = tables_dir / "layerwise_audit.csv"
    if not audit_path.exists():
        logger.error("Missing %s -- run scripts/run_interchange_eval.py first.", audit_path)
        return
    audit_df = pd.read_csv(audit_path)

    logger.info("Claim 1: probe-vs-intervention correlation")
    correlation = correlate_metrics(audit_df)
    correlation.to_csv(tables_dir / "claim1_correlation.csv", index=False)
    logger.info("\n%s", correlation.to_string(index=False))

    disagreement = ranking_disagreement(audit_df, "decodability", "css")
    disagreement.to_csv(tables_dir / "claim1_ranking_disagreement.csv", index=False)
    logger.info("Top ranking disagreements:\n%s", disagreement.head(10).to_string(index=False))

    _plot_layerwise_audit(audit_df, figures_dir / "layerwise_audit.png")
    logger.info("Wrote paper tables/figures to %s / %s", tables_dir, figures_dir)


def _plot_layerwise_audit(audit_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(audit_df["factor"].unique()), figsize=(6 * audit_df["factor"].nunique(), 4), squeeze=False)
    for ax, (factor, group) in zip(axes[0], audit_df.groupby("factor"), strict=False):
        for encoder, enc_group in group.groupby("encoder"):
            enc_group = enc_group.sort_values("layer")
            ax.plot(enc_group["layer"], enc_group["decodability"], marker="o", label=f"{encoder} decodability")
            ax.plot(enc_group["layer"], enc_group["css"], marker="s", linestyle="--", label=f"{encoder} CSS")
        ax.set_title(f"factor={factor}")
        ax.set_xlabel("layer")
        ax.set_ylabel("score")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
