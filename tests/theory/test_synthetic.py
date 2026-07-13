import numpy as np
import pytest
from scipy.linalg import subspace_angles

from isplit.theory.synthetic import (
    FactorSpec,
    build_orthogonal_mixing_model,
    intervene_latents,
    make_two_subspaces,
    sample_latents,
    synthesize,
)


def test_build_orthogonal_mixing_model_factors_are_orthogonal():
    factors = [FactorSpec("content", 4), FactorSpec("speaker", 3), FactorSpec("env", 2)]
    model = build_orthogonal_mixing_model(ambient_dim=20, factors=factors, seed=0)

    angles = subspace_angles(model.mixing["content"], model.mixing["speaker"])
    assert np.allclose(angles, np.pi / 2, atol=1e-8)


def test_build_orthogonal_mixing_model_rejects_oversized_factors():
    factors = [FactorSpec("a", 10), FactorSpec("b", 10)]
    with pytest.raises(ValueError):
        build_orthogonal_mixing_model(ambient_dim=15, factors=factors, seed=0)


def test_intervene_latents_changes_only_target_factor():
    factors = [FactorSpec("content", 3), FactorSpec("speaker", 2)]
    latents = sample_latents(factors, n=10, seed=0)
    new_latents = intervene_latents(latents, factor="speaker", seed=1)

    assert np.allclose(new_latents["content"], latents["content"])
    assert not np.allclose(new_latents["speaker"], latents["speaker"])
    assert new_latents["speaker"].shape == latents["speaker"].shape


def test_synthesize_shape_and_noise_scaling():
    factors = [FactorSpec("content", 3)]
    model = build_orthogonal_mixing_model(ambient_dim=10, factors=factors, seed=0)
    latents = sample_latents(factors, n=50, seed=1)

    h_low_noise = synthesize(model, latents, noise_std=1e-6, seed=2)
    h_high_noise = synthesize(model, latents, noise_std=5.0, seed=2)

    assert h_low_noise.shape == (50, 10)
    assert np.var(h_high_noise) > np.var(h_low_noise)


def test_make_two_subspaces_recovers_requested_principal_angle():
    for target_deg in (0.5, 15, 45, 89):
        u1, u2 = make_two_subspaces(ambient_dim=20, dim1=2, dim2=2, principal_angle_deg=target_deg, seed=0)
        angles_deg = np.degrees(subspace_angles(u1, u2))
        assert angles_deg.min() == pytest.approx(target_deg, abs=1e-3)


def test_make_two_subspaces_rejects_oversized_dims():
    with pytest.raises(ValueError):
        make_two_subspaces(ambient_dim=5, dim1=3, dim2=4, principal_angle_deg=30, seed=0)


def test_make_two_subspaces_orthonormal_bases():
    u1, u2 = make_two_subspaces(ambient_dim=15, dim1=3, dim2=3, principal_angle_deg=20, seed=0)
    assert np.allclose(u1.T @ u1, np.eye(3), atol=1e-8)
    assert np.allclose(u2.T @ u2, np.eye(3), atol=1e-8)
