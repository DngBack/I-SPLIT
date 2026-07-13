"""Interchange interventions (I-SPLIT core, section 5.3): the key causal test.

Given two representations h_a, h_b and an estimated factor subspace U_k,
swap factor k's component of h_a for h_b's:

    h_{a<-b}^(k) = h_a - P_k h_a + P_k h_b,   P_k = U_k U_k^+

A successful factor subspace should make downstream predictors for factor k
move from a's value toward b's (Transfer), while predictors for every other
factor/content stay close to a's original value (Preserve).
"""

import numpy as np

from isplit.utils.linalg import projector_from_basis


def swap_factor(h_a: np.ndarray, h_b: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Batched interchange intervention: h_{a<-b}^(k) = h_a - P h_a + P h_b.

    h_a, h_b: (D,) single vectors or (N, D) row-vector batches (same shape).
    basis: (D, R) estimated factor subspace (columns need not be orthonormal;
        the projector is built via pseudo-inverse so it's correct either way).

    P is symmetric (it's an orthogonal projector), so for row-vector batches
    H @ P gives the same per-row projection as P @ h for a column vector h.
    """
    if h_a.shape != h_b.shape:
        raise ValueError(f"h_a and h_b must have matching shape, got {h_a.shape} vs {h_b.shape}")
    p = projector_from_basis(basis)
    if h_a.ndim == 1:
        return h_a - p @ h_a + p @ h_b
    return h_a - h_a @ p.T + h_b @ p.T
