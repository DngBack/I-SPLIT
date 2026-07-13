"""Synthetic linear-mixing data generators used to validate the I-SPLIT theory
(Proposition 1: interventional identifiability; Proposition 2: separability without
orthogonality) before any real audio/encoder is involved.

Generative model: h = sum_k A_k z_k + noise, where each factor k has its own
mixing matrix A_k (ambient_dim x latent_dim_k) and latent code z_k.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FactorSpec:
    name: str
    latent_dim: int


@dataclass
class MixingModel:
    ambient_dim: int
    factors: list[FactorSpec]
    mixing: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (ambient_dim, latent_dim)


def build_orthogonal_mixing_model(
    ambient_dim: int, factors: list[FactorSpec], seed: int = 0
) -> MixingModel:
    """Build a mixing model whose factor subspaces are mutually orthogonal.

    Draws one random orthonormal basis for the full stacked dimension and
    slices it into per-factor blocks, so factors never overlap and are
    pairwise orthogonal by construction.
    """
    total_dim = sum(f.latent_dim for f in factors)
    if total_dim > ambient_dim:
        raise ValueError(
            f"sum of factor latent dims ({total_dim}) must be <= ambient_dim ({ambient_dim})"
        )
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((ambient_dim, total_dim)))
    mixing: dict[str, np.ndarray] = {}
    offset = 0
    for f in factors:
        mixing[f.name] = q[:, offset : offset + f.latent_dim]
        offset += f.latent_dim
    return MixingModel(ambient_dim=ambient_dim, factors=factors, mixing=mixing)


def sample_latents(
    factors: list[FactorSpec], n: int, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {f.name: rng.standard_normal((n, f.latent_dim)) for f in factors}


def synthesize(
    model: MixingModel, latents: dict[str, np.ndarray], noise_std: float, seed: int
) -> np.ndarray:
    """h = sum_k A_k z_k + noise."""
    n = next(iter(latents.values())).shape[0]
    rng = np.random.default_rng(seed)
    h = np.zeros((n, model.ambient_dim))
    for f in model.factors:
        h += latents[f.name] @ model.mixing[f.name].T
    h += noise_std * rng.standard_normal((n, model.ambient_dim))
    return h


def intervene_latents(
    latents: dict[str, np.ndarray], factor: str, seed: int
) -> dict[str, np.ndarray]:
    """Resample the latent code for exactly one factor, holding all others fixed.

    This is the synthetic analogue of a paired input intervention: only factor
    `factor`'s generative value changes between the "before" and "after" example.
    """
    rng = np.random.default_rng(seed)
    new_latents = dict(latents)
    dim = latents[factor].shape[1]
    n = latents[factor].shape[0]
    new_latents[factor] = rng.standard_normal((n, dim))
    return new_latents


def make_two_subspaces(
    ambient_dim: int, dim1: int, dim2: int, principal_angle_deg: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Construct two orthonormal-basis subspaces U1 (ambient_dim x dim1) and
    U2 (ambient_dim x dim2) whose *smallest* principal angle equals
    `principal_angle_deg` (all other principal angles are 90 degrees).

    Used to probe Proposition 2: separability holds whenever
    S_y ∩ S_n = {0} (angle > 0), independent of orthogonality (angle == 90),
    but reconstruction becomes ill-conditioned as the angle -> 0.
    """
    if ambient_dim < dim1 + dim2:
        raise ValueError("ambient_dim must be >= dim1 + dim2 for this construction")
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((ambient_dim, dim1 + dim2)))
    u1 = q[:, :dim1]
    u_perp = q[:, dim1 : dim1 + dim2]  # orthonormal, orthogonal to all of u1

    theta = np.deg2rad(principal_angle_deg)
    v0 = np.cos(theta) * u1[:, 0] + np.sin(theta) * u_perp[:, 0]

    if dim2 == 1:
        u2 = v0[:, None]
    else:
        u2 = np.column_stack([v0, u_perp[:, 1:dim2]])
        u2, _ = np.linalg.qr(u2)  # re-orthonormalize u2's own basis (span unchanged)

    return u1, u2
