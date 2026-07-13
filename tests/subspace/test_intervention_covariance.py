import numpy as np
import pytest

from isplit.causal.metrics import principal_angles
from isplit.subspace.intervention_covariance import (
    compute_deltas,
    eigendecompose,
    estimate_intervention_subspace,
    estimate_subspace_from_deltas,
    intervention_covariance,
    select_rank_by_energy,
)


def test_compute_deltas_is_b_minus_a():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[1.5, 2.5], [3.5, 4.5]])
    deltas = compute_deltas(a, b)
    assert np.allclose(deltas, [[0.5, 0.5], [0.5, 0.5]])


def test_compute_deltas_rejects_mismatched_shapes():
    a = np.zeros((3, 4))
    b = np.zeros((3, 5))
    with pytest.raises(ValueError):
        compute_deltas(a, b)


def test_intervention_covariance_recovers_known_low_rank_structure():
    rng = np.random.default_rng(0)
    d, r, n = 10, 2, 5000
    a_k, _ = np.linalg.qr(rng.standard_normal((d, r)))
    z = rng.standard_normal((n, r))
    deltas = z @ a_k.T  # no noise: covariance should be exactly rank r

    c = intervention_covariance(deltas)
    eigvals, _ = eigendecompose(c)

    assert eigvals[0] > 0
    assert np.all(eigvals[r:] < 1e-6 * eigvals[0])


def test_eigendecompose_sorted_descending():
    rng = np.random.default_rng(1)
    m = rng.standard_normal((5, 5))
    c = m @ m.T
    eigvals, _ = eigendecompose(c)
    assert np.all(np.diff(eigvals) <= 1e-10)


def test_select_rank_by_energy_picks_expected_rank():
    eigvals = np.array([10.0, 9.0, 0.05, 0.05, 0.05])
    rank = select_rank_by_energy(eigvals, energy_threshold=0.9)
    assert rank == 2


def test_select_rank_by_energy_all_zero_returns_one():
    assert select_rank_by_energy(np.zeros(5)) == 1


def test_estimate_subspace_recovers_ground_truth_as_n_grows():
    rng = np.random.default_rng(2)
    d, r = 30, 3
    a_k, _ = np.linalg.qr(rng.standard_normal((d, r)))

    errors = []
    for n in (20, 2000):
        z = rng.standard_normal((n, r))
        noise = 0.5 * rng.standard_normal((n, d))
        deltas = z @ a_k.T + noise
        u_hat, _, _ = estimate_subspace_from_deltas(deltas, rank=r)
        angles = principal_angles(a_k, u_hat, degrees=True)
        errors.append(np.mean(angles))

    assert errors[1] < errors[0]


def test_estimate_intervention_subspace_matches_manual_deltas():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((50, 8))
    b = a + rng.standard_normal((50, 8)) * 0.1
    u1, ev1, full1 = estimate_intervention_subspace(a, b, rank=2)
    u2, ev2, full2 = estimate_subspace_from_deltas(b - a, rank=2)
    assert np.allclose(u1, u2)
    assert np.allclose(ev1, ev2)
    assert np.allclose(full1, full2)


def test_intervention_covariance_rejects_empty():
    with pytest.raises(ValueError):
        intervention_covariance(np.zeros((0, 5)))
