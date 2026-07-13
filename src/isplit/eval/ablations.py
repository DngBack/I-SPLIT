"""Required ablations (paper section 12): true-vs-mismatched pairs, isolated-
vs-joint nuisance discovery, linear-vs-nonlinear leakage probes, and
gold-vs-pseudo pair labels. Rank/regularization sweep is `eval.pareto`; layer
sweep and multi-seed are orchestration (repeated calls into layerwise_audit /
this module across layers/seeds), not separate library logic.
"""

import numpy as np
import pandas as pd

from isplit.causal.metrics import principal_angles
from isplit.probes.linear import LinearProbe
from isplit.probes.mlp import MLPProbe
from isplit.subspace.intervention_covariance import estimate_subspace_from_deltas


def shuffle_pair_correspondence(pairs: pd.DataFrame, b_col: str, seed: int) -> pd.DataFrame:
    """Break the true pairing while preserving each column's marginal
    distribution: randomly permute the `b_col` values across rows. A subspace
    estimated from these mismatched pairs should recover much less structure
    than one from true pairs -- otherwise the method is just picking up
    ordinary class structure, not the intervention-specific signal.
    """
    rng = np.random.default_rng(seed)
    shuffled = pairs.copy()
    shuffled[b_col] = rng.permutation(shuffled[b_col].to_numpy())
    return shuffled


def true_vs_mismatched_subspace_quality(
    features_a: np.ndarray, features_b: np.ndarray, seed: int, rank: int
) -> dict[str, float]:
    """Compare the eigenvalue concentration (top-rank energy fraction) of the
    intervention covariance for true vs. row-shuffled (mismatched) pairs --
    true pairs should concentrate energy in far fewer dimensions.
    """
    deltas_true = features_b - features_a
    rng = np.random.default_rng(seed)
    shuffled_b = features_b[rng.permutation(len(features_b))]
    deltas_mismatched = shuffled_b - features_a

    _, eigvals_true, full_true = estimate_subspace_from_deltas(deltas_true, rank=rank)
    _, eigvals_mismatched, full_mismatched = estimate_subspace_from_deltas(deltas_mismatched, rank=rank)

    def _energy_fraction(top: np.ndarray, full: np.ndarray) -> float:
        total = np.clip(full, 0, None).sum()
        return float(np.clip(top, 0, None).sum() / total) if total > 0 else 0.0

    return {
        "true_top_rank_energy_fraction": _energy_fraction(eigvals_true, full_true),
        "mismatched_top_rank_energy_fraction": _energy_fraction(eigvals_mismatched, full_mismatched),
    }


def isolated_vs_joint_subspace_similarity(
    isolated_basis: np.ndarray, joint_basis: np.ndarray
) -> float:
    """Mean cos^2 of principal angles between a factor subspace estimated in
    isolation vs. estimated from pairs where multiple factors changed jointly
    -- tests whether factor subspaces compose (high similarity) or interfere
    (low similarity) when interventions aren't isolated.
    """
    angles = principal_angles(isolated_basis, joint_basis)
    if angles.size == 0:
        return 0.0
    return float(np.mean(np.cos(angles) ** 2))


def linear_vs_nonlinear_leakage(
    cleaned_features: np.ndarray, nuisance_labels: np.ndarray, seed: int = 0
) -> dict[str, float]:
    """A low linear-probe score doesn't prove erasure: fit both a linear and a
    small MLP probe for the nuisance label on the *cleaned* (projected)
    features and report both accuracies. A gap indicates nonlinear residual leakage.
    """
    n = len(cleaned_features)
    split = max(1, int(n * 0.8))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    train_idx, test_idx = idx[:split], idx[split:]
    if len(test_idx) == 0:
        return {"linear_leakage_acc": float("nan"), "nonlinear_leakage_acc": float("nan")}

    x_train, x_test = cleaned_features[train_idx], cleaned_features[test_idx]
    y_train, y_test = nuisance_labels[train_idx], nuisance_labels[test_idx]

    linear_acc = LinearProbe().fit(x_train, y_train).score(x_test, y_test)
    nonlinear_acc = MLPProbe(seed=seed).fit(x_train, y_train).score(x_test, y_test)
    return {"linear_leakage_acc": linear_acc, "nonlinear_leakage_acc": nonlinear_acc}


def pseudo_label_speaker_pairs(vctk_manifest: pd.DataFrame, n_pairs: int, seed: int) -> pd.DataFrame:
    """Gold-vs-pseudo ablation: construct "speaker-change" pairs *without*
    using the true prompt_id match (the gold signal), instead pairing
    utterances of similar duration as a cheap unsupervised proxy for "same
    content, different speaker." Expected to be a noisier signal than the
    gold VCTK prompt-matched pairs from `data.pairs.make_speaker_pairs`.
    """
    rng = np.random.default_rng(seed)
    df = vctk_manifest.copy()
    df["duration_proxy"] = df["text"].str.len()  # cheap proxy without loading audio
    df = df.sort_values("duration_proxy").reset_index(drop=True)

    rows = []
    attempts, max_attempts = 0, n_pairs * 30 + 100
    while len(rows) < n_pairs and attempts < max_attempts:
        attempts += 1
        i = rng.integers(0, len(df) - 1)
        a_row, b_row = df.iloc[i], df.iloc[i + 1]
        if a_row["speaker_id"] == b_row["speaker_id"]:
            continue
        rows.append(
            {
                "pair_id": len(rows),
                "factor": "speaker",
                "a_utt_id": a_row["utt_id"],
                "b_utt_id": b_row["utt_id"],
                "a_speaker_id": a_row["speaker_id"],
                "b_speaker_id": b_row["speaker_id"],
                "label_source": "pseudo",
            }
        )
    return pd.DataFrame(rows)
