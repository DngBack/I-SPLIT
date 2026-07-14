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


def _spearman(x, y) -> float:
    if len(x) < 3:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def cluster_bootstrap_spearman(
    audit_df: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    cluster_col: str = "encoder",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Spearman rho with a confidence interval from resampling whole *encoders*.

    The rows of the audit table are not independent observations: the 13 layers
    of one encoder are strongly autocorrelated, and every encoder contributes a
    block of them. A row-wise p-value therefore assumes far more independent
    evidence than exists and is anticonservative. Resampling entire encoders
    (a cluster bootstrap) keeps the within-encoder dependence intact, so the
    interval reflects how much the answer moves when the *encoders* change --
    which is the population the claim generalizes over.
    """
    valid = audit_df[[metric_a, metric_b, cluster_col]].dropna()
    rho = _spearman(valid[metric_a], valid[metric_b])
    clusters = valid[cluster_col].unique()
    if len(clusters) < 2 or np.isnan(rho):
        return {"spearman_rho": rho, "ci_lo": float("nan"), "ci_hi": float("nan"), "n_clusters": len(clusters), "n": len(valid)}

    rng = np.random.default_rng(seed)
    groups = {c: valid[valid[cluster_col] == c] for c in clusters}
    boots = []
    for _ in range(n_boot):
        drawn = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([groups[c] for c in drawn])
        r = _spearman(sample[metric_a], sample[metric_b])
        if not np.isnan(r):
            boots.append(r)
    if not boots:
        return {"spearman_rho": rho, "ci_lo": float("nan"), "ci_hi": float("nan"), "n_clusters": len(clusters), "n": len(valid)}

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "spearman_rho": rho,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_clusters": int(len(clusters)),
        "n": int(len(valid)),
    }


def baseline_comparison(
    audit_df: pd.DataFrame,
    value_col: str = "css",
    reference: str = "isplit",
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Per factor: mean CSS of each subspace method, and the paired advantage of
    `reference` over each baseline with a cluster-bootstrap CI over encoders.

    This is the control the causal claim rests on. CSS is not calibrated in the
    abstract -- an interchange on a *random* subspace of the same rank already
    moves a probe some of the time, so "I-SPLIT reaches CSS 0.72" is only
    meaningful next to what PCA and a random subspace reach on the identical
    pairs. Deltas are paired within (encoder, layer, factor) so the comparison
    is not confounded by layer difficulty.
    """
    keys = ["encoder", "layer", "factor"]
    wide = audit_df.pivot_table(index=keys, columns="method", values=value_col).reset_index()
    if reference not in wide.columns:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows = []
    for factor, group in wide.groupby("factor"):
        encoders = group["encoder"].unique()
        for method in [m for m in wide.columns if m not in keys and m != reference]:
            paired = group[[reference, method, "encoder"]].dropna()
            if paired.empty:
                continue
            delta = paired[reference] - paired[method]
            boots = []
            if len(encoders) >= 2:
                by_enc = {e: delta[paired["encoder"] == e] for e in encoders}
                for _ in range(n_boot):
                    drawn = rng.choice(encoders, size=len(encoders), replace=True)
                    boots.append(float(pd.concat([by_enc[e] for e in drawn]).mean()))
            lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
            rows.append(
                {
                    "factor": factor,
                    "method": method,
                    f"mean_{value_col}": float(paired[method].mean()),
                    f"mean_{value_col}_{reference}": float(paired[reference].mean()),
                    "delta_vs_reference": float(delta.mean()),
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                    "beats_baseline": bool(lo > 0) if not np.isnan(lo) else False,
                    "n": int(len(paired)),
                }
            )
    return pd.DataFrame(rows).sort_values(["factor", "method"]).reset_index(drop=True)


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
