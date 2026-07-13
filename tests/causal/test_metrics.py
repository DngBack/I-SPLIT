import numpy as np
import pytest
from scipy.linalg import subspace_angles

from isplit.causal.metrics import (
    causal_selectivity_score,
    classification_preserve,
    classification_transfer,
    clipped_score,
    principal_angles,
    probability_divergence,
)


def test_clipped_score_basic():
    assert clipped_score(0.0) == 1.0
    assert clipped_score(1.0) == 0.0
    assert clipped_score(2.0) == 0.0  # clipped, not negative


def test_classification_preserve_and_transfer():
    assert classification_preserve("a", "a") == 1.0
    assert classification_preserve("a", "b") == 0.0
    assert classification_transfer("b", "b") == 1.0
    assert classification_transfer("a", "b") == 0.0


def test_probability_divergence_identical_is_zero():
    p = np.array([0.2, 0.3, 0.5])
    assert probability_divergence(p, p) == pytest.approx(0.0, abs=1e-8)


def test_probability_divergence_disjoint_supports_is_one():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert probability_divergence(p, q) == pytest.approx(1.0, abs=1e-6)


def test_causal_selectivity_score_harmonic_mean():
    assert causal_selectivity_score(1.0, 1.0) == pytest.approx(1.0)
    assert causal_selectivity_score(0.5, 0.5) == pytest.approx(0.5)
    assert causal_selectivity_score(0.0, 1.0) == 0.0
    assert causal_selectivity_score(1.0, 0.0) == 0.0


def test_causal_selectivity_score_penalizes_imbalance():
    # harmonic mean of (0.1, 0.9) should be much closer to 0.1 than the arithmetic mean (0.5)
    css = causal_selectivity_score(0.1, 0.9)
    assert css < 0.5


def test_principal_angles_matches_scipy_reference():
    rng = np.random.default_rng(0)
    u1, _ = np.linalg.qr(rng.standard_normal((10, 3)))
    u2, _ = np.linalg.qr(rng.standard_normal((10, 3)))
    expected = subspace_angles(u1, u2)
    got = principal_angles(u1, u2)
    assert np.allclose(got, expected)


def test_principal_angles_identical_subspace_is_zero():
    rng = np.random.default_rng(1)
    u, _ = np.linalg.qr(rng.standard_normal((10, 4)))
    angles = principal_angles(u, u)
    assert np.allclose(angles, 0.0, atol=1e-8)


def test_principal_angles_orthogonal_subspace_is_90deg():
    d = 10
    u1 = np.eye(d)[:, :3]
    u2 = np.eye(d)[:, 3:6]
    angles = principal_angles(u1, u2, degrees=True)
    assert np.allclose(angles, 90.0, atol=1e-6)
