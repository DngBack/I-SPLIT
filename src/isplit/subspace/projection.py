"""Regularized oblique projection (I-SPLIT core, section 5.2 / Proposition 2).

Instead of assuming factor subspaces must be orthogonal, stack the estimated
per-factor bases B = [U_1 | U_2 | ...] and decompose a representation h via
ridge-regularized least squares:

    a_hat = (B^T B + tau I)^-1 B^T h

then reconstruct only the coefficient block(s) belonging to the factor(s) you
want to keep, zeroing the rest. This is the single highest bug-risk module in
the codebase (a silent transpose/pseudo-inverse error here produces
plausible-but-wrong results), so every function here is covered by tests
against analytically-known synthetic ground truth.
"""

import numpy as np

from isplit.utils.linalg import orthonormalize, projector_from_basis


def fit_oblique(basis: np.ndarray, features: np.ndarray, tau: float) -> np.ndarray:
    """Ridge-regularized least-squares coefficients for a stacked basis.

    basis: (D, R) stacked factor bases [U_1 | U_2 | ...]
    features: (N, D) representations, one per row
    tau: ridge regularization strength (>0 keeps this well-posed even when
        `basis` is ill-conditioned, e.g. near-parallel factor subspaces)

    Returns A_hat: (N, R) per-example coefficients, block-indexed in the same
    column order as `basis`.
    """
    if features.ndim == 1:
        features = features[None, :]
    r = basis.shape[1]
    gram = basis.T @ basis + tau * np.eye(r)
    rhs = basis.T @ features.T  # (R, N)
    a_hat = np.linalg.solve(gram, rhs).T  # (N, R)
    return a_hat


def reconstruct_block(basis: np.ndarray, a_hat: np.ndarray, block: slice) -> np.ndarray:
    """Reconstruct representations using only the coefficients/basis columns in
    `block` (e.g. the content block), implicitly zeroing every other block.

    Returns (N, D).
    """
    basis_block = basis[:, block]
    a_block = a_hat[:, block]
    return a_block @ basis_block.T


def oblique_reconstruct(
    basis: np.ndarray,
    features: np.ndarray,
    tau: float,
    keep_block: slice,
) -> np.ndarray:
    """Convenience wrapper: fit oblique coefficients then reconstruct only
    `keep_block` (e.g. slice(0, content_rank) to keep content and drop nuisance).
    """
    a_hat = fit_oblique(basis, features, tau)
    return reconstruct_block(basis, a_hat, keep_block)


def orthogonal_projection_remove(features: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Baseline: remove the component of `features` lying in col(basis) via an
    orthogonal projector (nullspace removal), regardless of whether `basis`'s
    own columns are orthonormal.
    """
    p = projector_from_basis(basis)
    return features - features @ p.T


def orthogonal_projection_keep(features: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Baseline: keep only the component of `features` lying in col(basis)."""
    p = projector_from_basis(basis)
    return features @ p.T


def random_subspace(ambient_dim: int, rank: int, seed: int) -> np.ndarray:
    """Control baseline: an orthonormal random subspace of the given rank,
    matched in dimensionality to a real estimated factor subspace.
    """
    rng = np.random.default_rng(seed)
    return orthonormalize(rng.standard_normal((ambient_dim, rank)))
