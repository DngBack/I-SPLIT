import numpy as np
import pytest

from isplit.subspace.projection import (
    fit_oblique,
    oblique_reconstruct,
    orthogonal_projection_keep,
    orthogonal_projection_remove,
    random_subspace,
    reconstruct_block,
)
from isplit.theory.synthetic import make_two_subspaces


def test_oblique_recovers_true_coefficients_no_noise():
    rng = np.random.default_rng(0)
    d, r, n = 20, 5, 200
    basis, _ = np.linalg.qr(rng.standard_normal((d, r)))
    a_true = rng.standard_normal((n, r))
    h = a_true @ basis.T

    a_hat = fit_oblique(basis, h, tau=1e-10)

    assert np.allclose(a_hat, a_true, atol=1e-6)


def test_oblique_zeroes_nuisance_block_exactly_when_orthogonal_and_untainted():
    rng = np.random.default_rng(1)
    d, content_rank, nuisance_rank, n = 20, 3, 2, 100
    q, _ = np.linalg.qr(rng.standard_normal((d, content_rank + nuisance_rank)))
    u_y, u_n = q[:, :content_rank], q[:, content_rank:]
    basis = np.concatenate([u_y, u_n], axis=1)

    a_true = rng.standard_normal((n, content_rank + nuisance_rank))
    h = a_true @ basis.T
    content_true = a_true[:, :content_rank] @ u_y.T

    content_only = oblique_reconstruct(basis, h, tau=1e-10, keep_block=slice(0, content_rank))

    assert np.allclose(content_only, content_true, atol=1e-6)


def test_oblique_graceful_near_singular_basis_does_not_blow_up():
    u1, u2 = make_two_subspaces(ambient_dim=20, dim1=3, dim2=3, principal_angle_deg=0.001, seed=0)
    basis = np.concatenate([u1, u2], axis=1)
    rng = np.random.default_rng(2)
    h = rng.standard_normal((10, 20))

    a_hat = fit_oblique(basis, h, tau=1e-3)

    assert np.all(np.isfinite(a_hat))


def test_oblique_zero_tau_near_singular_basis_can_blow_up_or_stay_finite():
    # Documents the failure mode tau is meant to guard against: with tau=0 and a
    # near-parallel basis, the Gram matrix is nearly singular. We only assert this
    # doesn't raise (np.linalg.solve on a near-singular but non-exactly-singular
    # matrix still returns a finite, if large/inaccurate, result).
    u1, u2 = make_two_subspaces(ambient_dim=20, dim1=2, dim2=2, principal_angle_deg=0.0001, seed=0)
    basis = np.concatenate([u1, u2], axis=1)
    rng = np.random.default_rng(3)
    h = rng.standard_normal((5, 20))

    a_hat = fit_oblique(basis, h, tau=0.0)

    assert a_hat.shape == (5, 4)


def test_orthogonal_projection_remove_then_keep_reconstructs_original():
    rng = np.random.default_rng(4)
    d, rank, n = 15, 3, 50
    basis = rng.standard_normal((d, rank))  # deliberately non-orthonormal columns
    h = rng.standard_normal((n, d))

    removed = orthogonal_projection_remove(h, basis)
    kept = orthogonal_projection_keep(h, basis)

    assert np.allclose(removed + kept, h, atol=1e-8)


def test_orthogonal_projection_remove_is_idempotent():
    rng = np.random.default_rng(5)
    basis = rng.standard_normal((10, 3))
    h = rng.standard_normal((20, 10))

    once = orthogonal_projection_remove(h, basis)
    twice = orthogonal_projection_remove(once, basis)

    assert np.allclose(once, twice, atol=1e-8)


def test_random_subspace_is_orthonormal_and_correct_shape():
    u = random_subspace(ambient_dim=30, rank=4, seed=0)
    assert u.shape == (30, 4)
    assert np.allclose(u.T @ u, np.eye(4), atol=1e-8)


def test_reconstruct_block_matches_manual_slice():
    rng = np.random.default_rng(6)
    basis = rng.standard_normal((10, 6))
    a_hat = rng.standard_normal((4, 6))
    block = slice(2, 5)

    out = reconstruct_block(basis, a_hat, block)
    manual = a_hat[:, block] @ basis[:, block].T

    assert np.allclose(out, manual)


def test_fit_oblique_rejects_mismatched_feature_dim():
    rng = np.random.default_rng(7)
    basis = rng.standard_normal((10, 3))
    h_wrong_dim = rng.standard_normal((5, 11))
    with pytest.raises(ValueError):
        fit_oblique(basis, h_wrong_dim, tau=0.1)
