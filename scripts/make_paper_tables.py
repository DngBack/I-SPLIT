"""Assemble the paper's core tables/figures from the outputs of the earlier
scripts. Run after extract_features.py -> fit_subspaces.py -> run_interchange_eval.py.

Claim 1 is tested here on three fronts:

* **against a control** -- CSS for the I-SPLIT subspace vs. PCA and a random
  subspace of the same rank, paired within (encoder, layer, factor). An
  interchange on a random subspace already moves a probe some of the time, so
  an uncontrolled CSS has no scale.
* **on a decodability measure that has variance** -- prompt-id accuracy pins at
  1.0 on most layers, and a metric with no variance cannot correlate with
  anything, so content is also read through the CTC probe's CER.
* **with statistics that respect the data's structure** -- rho is reported with
  a cluster bootstrap over encoders, because the 13 layers of one encoder are
  not 13 independent observations and a row-wise p-value pretends they are.

Usage: uv run python scripts/make_paper_tables.py --scale pilot
"""

from pathlib import Path

import click
import numpy as np
import pandas as pd

from isplit.config.loader import load_config
from isplit.eval.probe_vs_intervention import (
    baseline_comparison,
    cluster_bootstrap_spearman,
    ranking_disagreement,
)
from isplit.utils.logging import get_logger

logger = get_logger(__name__)

REFERENCE_METHOD = "isplit"


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

    # ---- Control: does the I-SPLIT subspace actually beat PCA / random? -------
    baselines = baseline_comparison(audit_df, value_col="css", reference=REFERENCE_METHOD)
    baselines.to_csv(tables_dir / "baseline_comparison.csv", index=False)
    logger.info("CSS vs. baselines (paired, 95%% CI bootstrapped over encoders):\n%s",
                baselines.to_string(index=False))

    # ---- Claim 1: does decodability / geometry predict causal selectivity? ----
    isplit_df = audit_df[audit_df["method"] == REFERENCE_METHOD].copy()
    # a non-saturating content decodability: 1 - CER
    isplit_df["cer_decodability"] = 1.0 - isplit_df["content_cer"]

    rows = []
    for metric_a, metric_b, subset in [
        ("decodability", "css", None),
        ("iei", "css", None),
        ("decodability", "iei", None),
        ("cer_decodability", "css", "content"),
        ("decodability", "css", "content"),
        ("decodability", "css", "speaker"),
        ("decodability", "css", "environment"),
        ("decodability", "css", "channel"),
    ]:
        frame = isplit_df if subset is None else isplit_df[isplit_df["factor"] == subset]
        if metric_a not in frame.columns or frame[[metric_a, metric_b]].dropna().empty:
            continue
        stats = cluster_bootstrap_spearman(frame, metric_a, metric_b)
        rows.append({"factor": subset or "all", "metric_a": metric_a, "metric_b": metric_b, **stats})
    correlation = pd.DataFrame(rows)
    correlation.to_csv(tables_dir / "claim1_correlation.csv", index=False)
    logger.info("Claim 1 correlations (95%% CI over encoders):\n%s", correlation.to_string(index=False))

    # Is the decodability measure even capable of correlating with anything?
    ceiling = (
        isplit_df.groupby("factor")["decodability"]
        .agg(mean="mean", std="std", frac_at_ceiling=lambda s: float((s >= 0.99).mean()))
        .reset_index()
    )
    ceiling.to_csv(tables_dir / "decodability_ceiling.csv", index=False)
    logger.info("Decodability variance / ceiling check:\n%s", ceiling.to_string(index=False))

    disagreement = ranking_disagreement(isplit_df, "decodability", "css")
    disagreement.to_csv(tables_dir / "claim1_ranking_disagreement.csv", index=False)

    _plot_layerwise_audit(isplit_df, figures_dir / "layerwise_audit.png")
    _plot_baselines(audit_df, figures_dir / "css_vs_baselines.png")
    logger.info("Wrote paper tables/figures to %s / %s", tables_dir, figures_dir)


def _plot_layerwise_audit(audit_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    factors = sorted(audit_df["factor"].unique())
    fig, axes = plt.subplots(1, len(factors), figsize=(5.5 * len(factors), 4), squeeze=False)
    for ax, factor in zip(axes[0], factors, strict=False):
        group = audit_df[audit_df["factor"] == factor]
        for encoder, enc_group in group.groupby("encoder"):
            enc_group = enc_group.sort_values("layer")
            ax.plot(enc_group["layer"], enc_group["decodability"], marker="o", label=f"{encoder} decodability")
            ax.plot(enc_group["layer"], enc_group["css"], marker="s", linestyle="--", label=f"{encoder} CSS")
        ax.set_title(f"factor={factor}")
        ax.set_xlabel("layer")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_baselines(audit_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    factors = sorted(audit_df["factor"].unique())
    methods = ["isplit", "pca", "random"]
    fig, axes = plt.subplots(1, len(factors), figsize=(5.5 * len(factors), 4), squeeze=False)
    for ax, factor in zip(axes[0], factors, strict=False):
        group = audit_df[audit_df["factor"] == factor]
        for method in methods:
            m = group[group["method"] == method].groupby("layer")["css"].mean()
            ax.plot(m.index, m.values, marker="o", label=method)
        ax.set_title(f"CSS vs. baselines -- {factor}")
        ax.set_xlabel("layer")
        ax.set_ylabel("CSS (mean over encoders)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
