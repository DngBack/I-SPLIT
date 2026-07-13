"""Small numerically-careful linear algebra helpers shared across subspace/causal modules."""

import numpy as np


def safe_pinv(a: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    return np.linalg.pinv(a, rcond=rcond)


def orthonormalize(u: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis (via QR) spanning the same column space as u.

    Note: this changes the individual basis vectors (not just their span), so it
    must not be used where the *specific* (non-orthonormal) directions in u matter
    -- only where "some orthonormal basis for span(u)" is what's needed.
    """
    if u.shape[1] == 0:
        return u
    q, _ = np.linalg.qr(u)
    return q


def projector_from_basis(u: np.ndarray) -> np.ndarray:
    """Orthogonal projector P = U U^+ onto col(U).

    Uses the Moore-Penrose pseudo-inverse so this is correct regardless of
    whether U's columns are orthonormal: for any full-column-rank U,
    U (U^T U)^-1 U^T == U U^+ is exactly the orthogonal projector onto col(U).
    Handles the degenerate zero-column case (empty subspace) by returning the
    zero matrix, i.e. a no-op projector.
    """
    d = u.shape[0]
    if u.shape[1] == 0:
        return np.zeros((d, d))
    u_pinv = safe_pinv(u)
    return u @ u_pinv


def relative_error(estimate: np.ndarray, target: np.ndarray) -> float:
    num = np.linalg.norm(estimate - target)
    den = np.linalg.norm(target) + 1e-12
    return float(num / den)
