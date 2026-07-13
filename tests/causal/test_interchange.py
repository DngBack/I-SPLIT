import numpy as np
import pytest

from isplit.causal.interchange import swap_factor
from isplit.utils.linalg import projector_from_basis


def test_swap_self_is_noop():
    rng = np.random.default_rng(0)
    h_a = rng.standard_normal(10)
    basis = rng.standard_normal((10, 3))
    out = swap_factor(h_a, h_a, basis)
    assert np.allclose(out, h_a, atol=1e-10)


def test_swap_full_ambient_basis_is_full_replacement():
    rng = np.random.default_rng(1)
    d = 8
    h_a = rng.standard_normal(d)
    h_b = rng.standard_normal(d)
    basis = np.eye(d)  # spans the full ambient space -> projector is identity
    out = swap_factor(h_a, h_b, basis)
    assert np.allclose(out, h_b, atol=1e-10)


def test_swap_zero_dim_basis_is_noop():
    rng = np.random.default_rng(2)
    h_a = rng.standard_normal(6)
    h_b = rng.standard_normal(6)
    basis = np.zeros((6, 0))
    out = swap_factor(h_a, h_b, basis)
    assert np.allclose(out, h_a, atol=1e-10)


def test_swap_matches_explicit_projector_formula_orthonormal_basis():
    rng = np.random.default_rng(3)
    d, r = 12, 4
    basis, _ = np.linalg.qr(rng.standard_normal((d, r)))
    h_a = rng.standard_normal(d)
    h_b = rng.standard_normal(d)

    out = swap_factor(h_a, h_b, basis)
    p = basis @ basis.T  # exact orthogonal projector for an orthonormal basis
    expected = h_a - p @ h_a + p @ h_b

    assert np.allclose(out, expected, atol=1e-10)


def test_swap_batched_matches_per_row_single_vector_swap():
    rng = np.random.default_rng(4)
    n, d, r = 5, 10, 3
    basis = rng.standard_normal((d, r))
    h_a = rng.standard_normal((n, d))
    h_b = rng.standard_normal((n, d))

    batched = swap_factor(h_a, h_b, basis)
    for i in range(n):
        single = swap_factor(h_a[i], h_b[i], basis)
        assert np.allclose(batched[i], single, atol=1e-10)


def test_swap_nonorthonormal_basis_uses_true_orthogonal_projector():
    rng = np.random.default_rng(5)
    d = 10
    # non-orthonormal basis spanning a 3-dim subspace
    basis = rng.standard_normal((d, 3)) * 3.7
    h_a = rng.standard_normal(d)
    h_b = rng.standard_normal(d)

    out = swap_factor(h_a, h_b, basis)
    p = projector_from_basis(basis)
    assert np.allclose(p, p.T, atol=1e-8)  # projector must be symmetric
    assert np.allclose(p @ p, p, atol=1e-6)  # projector must be idempotent
    expected = h_a - p @ h_a + p @ h_b
    assert np.allclose(out, expected, atol=1e-8)


def test_swap_rejects_shape_mismatch():
    rng = np.random.default_rng(6)
    h_a = rng.standard_normal(5)
    h_b = rng.standard_normal(6)
    basis = rng.standard_normal((5, 2))
    with pytest.raises(ValueError):
        swap_factor(h_a, h_b, basis)
