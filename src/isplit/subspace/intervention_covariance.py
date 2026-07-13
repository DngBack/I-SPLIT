"""Intervention-covariance subspace estimation (I-SPLIT core, section 5.1 / Proposition 1).

For a factor k, given paired examples (x_i, x_i^(k)) that differ only in factor k,
the intervention differences Delta_i = E(x_i^(k)) - E(x_i) concentrate their
covariance mass along the factor's mixing directions:

    C_k = mean_i Delta_i Delta_i^T  ~=  A_k Sigma_k A_k^T + C_noise

so the dominant eigenspace of C_k estimates S_k = col(A_k), up to rotation.
"""

import numpy as np


def compute_deltas(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    """Delta_i = E(x_i^(k)) - E(x_i), paired row-wise. Shapes (N, D) -> (N, D)."""
    if features_a.shape != features_b.shape:
        raise ValueError(
            f"paired features must have matching shape, got {features_a.shape} vs {features_b.shape}"
        )
    return features_b - features_a


def intervention_covariance(deltas: np.ndarray, center: bool = False) -> np.ndarray:
    """C_k = (1/N) sum_i Delta_i Delta_i^T.

    center=False by default: the raw second moment is what the theory calls
    for (direction of the intervention effect matters, not just its spread
    around the mean intervention effect). Set center=True to instead estimate
    the covariance of Delta about its mean, which is only preferable if you
    specifically want to ignore a systematic (average) intervention offset.
    """
    if deltas.ndim != 2:
        raise ValueError(f"deltas must be 2D (N, D), got shape {deltas.shape}")
    if center:
        deltas = deltas - deltas.mean(axis=0, keepdims=True)
    n = deltas.shape[0]
    if n == 0:
        raise ValueError("need at least one paired example to estimate a covariance")
    return (deltas.T @ deltas) / n


def eigendecompose(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition sorted by descending eigenvalue."""
    eigvals, eigvecs = np.linalg.eigh(c)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def select_rank_by_energy(eigvals_sorted: np.ndarray, energy_threshold: float = 0.95) -> int:
    """Smallest rank r such that the top-r eigenvalues capture >= energy_threshold
    of total (non-negative-clipped) eigenvalue mass. Used instead of a hand-picked
    fixed rank, per the paper's "effective rank via held-out intervention energy" design.
    """
    clipped = np.clip(eigvals_sorted, 0.0, None)
    total = clipped.sum()
    if total <= 0:
        return 1
    cumulative = np.cumsum(clipped) / total
    return int(np.searchsorted(cumulative, energy_threshold) + 1)


def estimate_subspace_from_deltas(
    deltas: np.ndarray,
    rank: int | None = None,
    energy_threshold: float = 0.95,
    center: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """End-to-end: deltas -> covariance -> eigendecomposition -> rank selection.

    Returns (U, eigvals_used, eigvals_full) where U is (D, rank) with
    orthonormal columns (eigenvectors of a symmetric matrix), eigvals_used are
    its top-`rank` eigenvalues, and eigvals_full is the complete sorted
    spectrum (useful for diagnostics / energy plots).
    """
    c = intervention_covariance(deltas, center=center)
    eigvals_sorted, eigvecs_sorted = eigendecompose(c)
    if rank is None:
        rank = select_rank_by_energy(eigvals_sorted, energy_threshold)
    rank = max(1, min(rank, eigvecs_sorted.shape[1]))
    u = eigvecs_sorted[:, :rank]
    return u, eigvals_sorted[:rank], eigvals_sorted


def estimate_intervention_subspace(
    features_a: np.ndarray,
    features_b: np.ndarray,
    rank: int | None = None,
    energy_threshold: float = 0.95,
    center: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper: paired raw features -> estimated factor subspace."""
    deltas = compute_deltas(features_a, features_b)
    return estimate_subspace_from_deltas(
        deltas, rank=rank, energy_threshold=energy_threshold, center=center
    )
