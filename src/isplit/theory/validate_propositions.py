"""Empirical validation of the two theoretical claims underlying I-SPLIT,
using only synthetic linear-mixing data (no audio/encoders needed):

Proposition 1 (interventional identifiability): the dominant eigenspace of the
intervention covariance C_k recovers the true factor subspace col(A_k), with
recovery error (principal angle) shrinking as sample size N grows, as
intervention strength grows relative to noise, and getting worse as noise grows.

Proposition 2 (separability without orthogonality): the regularized oblique
projector can cleanly separate content from nuisance whenever the two
subspaces are merely disjoint (principal angle > 0), not just when they're
orthogonal (angle == 90); the reconstruction becomes ill-conditioned as the
smallest principal angle -> 0, which the ridge term tau is meant to control.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from isplit.causal.metrics import principal_angles
from isplit.subspace.intervention_covariance import estimate_subspace_from_deltas
from isplit.subspace.projection import fit_oblique, reconstruct_block
from isplit.theory.synthetic import make_two_subspaces
from isplit.utils.linalg import relative_error


def prop1_eigenspace_identifiability(
    ambient_dim: int = 64,
    factor_rank: int = 4,
    noise_std: float = 0.3,
    intervention_strength: float = 1.0,
    n_values: Sequence[int] = (10, 30, 100, 300, 1000, 3000),
    seed: int = 0,
) -> pd.DataFrame:
    """Sweep sample size N; for each N, estimate the factor subspace from
    synthetic intervention deltas and measure its principal-angle recovery
    error against the known ground-truth mixing matrix A_k.
    """
    rng = np.random.default_rng(seed)
    a_k, _ = np.linalg.qr(rng.standard_normal((ambient_dim, factor_rank)))

    rows = []
    for n in n_values:
        z_before = rng.standard_normal((n, factor_rank))
        z_after = rng.standard_normal((n, factor_rank))
        delta_z = intervention_strength * (z_after - z_before)
        deltas_clean = delta_z @ a_k.T
        noise = noise_std * rng.standard_normal((n, ambient_dim))
        deltas = deltas_clean + noise

        u_hat, _, _ = estimate_subspace_from_deltas(deltas, rank=factor_rank)
        angles_deg = principal_angles(a_k, u_hat, degrees=True)
        rows.append(
            {
                "n": n,
                "mean_angle_deg": float(np.mean(angles_deg)),
                "max_angle_deg": float(np.max(angles_deg)),
            }
        )
    return pd.DataFrame(rows)


def prop1_noise_sensitivity(
    ambient_dim: int = 64,
    factor_rank: int = 4,
    n: int = 500,
    intervention_strength: float = 1.0,
    noise_std_values: Sequence[float] = (0.01, 0.1, 0.3, 1.0, 3.0),
    seed: int = 0,
) -> pd.DataFrame:
    """Sweep noise std at fixed N; recovery error should grow with noise
    (equivalently, shrink as the effective SNR between intervention strength
    and noise grows) -- the other half of the identifiability claim.
    """
    rng = np.random.default_rng(seed)
    a_k, _ = np.linalg.qr(rng.standard_normal((ambient_dim, factor_rank)))

    rows = []
    for noise_std in noise_std_values:
        z_before = rng.standard_normal((n, factor_rank))
        z_after = rng.standard_normal((n, factor_rank))
        delta_z = intervention_strength * (z_after - z_before)
        deltas_clean = delta_z @ a_k.T
        noise = noise_std * rng.standard_normal((n, ambient_dim))
        deltas = deltas_clean + noise

        u_hat, _, _ = estimate_subspace_from_deltas(deltas, rank=factor_rank)
        angles_deg = principal_angles(a_k, u_hat, degrees=True)
        rows.append({"noise_std": noise_std, "mean_angle_deg": float(np.mean(angles_deg))})
    return pd.DataFrame(rows)


def prop2_oblique_vs_orthogonal(
    ambient_dim: int = 64,
    content_rank: int = 4,
    nuisance_rank: int = 4,
    n: int = 500,
    noise_std: float = 0.1,
    angles_deg: Sequence[float] = (90, 60, 30, 15, 5, 1),
    tau: float = 1e-2,
    seed: int = 0,
) -> pd.DataFrame:
    """Sweep the principal angle between a content and a nuisance subspace.
    Compares orthogonal-projection removal against regularized oblique
    reconstruction, both measured by relative error to the true (noise-free)
    content signal, alongside the basis's condition number.
    """
    from isplit.subspace.projection import orthogonal_projection_remove

    rng = np.random.default_rng(seed)
    rows = []
    for angle in angles_deg:
        u_y, u_n = make_two_subspaces(ambient_dim, content_rank, nuisance_rank, angle, seed=seed)
        z_y = rng.standard_normal((n, content_rank))
        z_n = rng.standard_normal((n, nuisance_rank))
        h_clean = z_y @ u_y.T + z_n @ u_n.T
        h = h_clean + noise_std * rng.standard_normal((n, ambient_dim))
        content_true = z_y @ u_y.T

        h_orth = orthogonal_projection_remove(h, u_n)
        orth_err = relative_error(h_orth, content_true)

        basis = np.concatenate([u_y, u_n], axis=1)
        a_hat = fit_oblique(basis, h, tau=tau)
        h_obl = reconstruct_block(basis, a_hat, slice(0, content_rank))
        obl_err = relative_error(h_obl, content_true)

        rows.append(
            {
                "principal_angle_deg": angle,
                "orthogonal_content_error": orth_err,
                "oblique_content_error": obl_err,
                "condition_number_basis": float(np.linalg.cond(basis)),
            }
        )
    return pd.DataFrame(rows)
