"""Claim 1: correlate probe-based decodability and geometric orthogonality
against intervention-based causal selectivity, across layers/encoders/factors.
The central finding this module is built to test: these metrics can rank
layers/models differently, i.e. probe accuracy is not a reliable stand-in for
causal separability.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def correlate_metrics(audit_df: pd.DataFrame) -> pd.DataFrame:
    """audit_df: one row per (encoder, layer, factor) with columns
    'decodability', 'iei' (geometric orthogonality proxy), 'css' (causal
    selectivity). Returns Spearman correlations between each pair of metrics,
    both overall and per-encoder (rank correlation, since what matters is
    whether these metrics *rank* layers/models the same way, not their raw scale).
    """
    pairs = [("decodability", "css"), ("iei", "css"), ("decodability", "iei")]
    rows = []
    for metric_a, metric_b in pairs:
        valid = audit_df[[metric_a, metric_b]].dropna()
        if len(valid) < 3:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = spearmanr(valid[metric_a], valid[metric_b])
        rows.append({"metric_a": metric_a, "metric_b": metric_b, "spearman_rho": rho, "p_value": p, "n": len(valid)})
    return pd.DataFrame(rows)


def ranking_disagreement(audit_df: pd.DataFrame, metric_a: str, metric_b: str) -> pd.DataFrame:
    """For each encoder, rank layers by metric_a and by metric_b and report
    the layers where the two rankings disagree most (largest rank-position
    gap) -- concrete evidence that a layer can look good on one metric and
    bad on the other.
    """
    rows = []
    for encoder, group in audit_df.groupby("encoder"):
        g = group.dropna(subset=[metric_a, metric_b]).copy()
        if len(g) < 2:
            continue
        g["rank_a"] = g[metric_a].rank(ascending=False)
        g["rank_b"] = g[metric_b].rank(ascending=False)
        g["rank_gap"] = (g["rank_a"] - g["rank_b"]).abs()
        rows.append(g[["encoder", "layer", "factor", metric_a, metric_b, "rank_a", "rank_b", "rank_gap"]])
    if not rows:
        return pd.DataFrame(columns=["encoder", "layer", "factor", metric_a, metric_b, "rank_a", "rank_b", "rank_gap"])
    return pd.concat(rows, ignore_index=True).sort_values("rank_gap", ascending=False)
