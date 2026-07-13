import numpy as np

from isplit.subspace.baselines import (
    class_centroid_pca,
    low_rank_svd_factorization,
    plain_pca,
    probe_nullspace_removal,
    random_subspace_baseline,
)


def test_plain_pca_recovers_known_low_rank_structure():
    rng = np.random.default_rng(0)
    d, r, n = 30, 3, 500
    basis, _ = np.linalg.qr(rng.standard_normal((d, r)))
    z = rng.standard_normal((n, r))
    features = z @ basis.T + 0.01 * rng.standard_normal((n, d))

    u = plain_pca(features, rank=r)
    assert u.shape == (d, r)
    assert np.allclose(u.T @ u, np.eye(r), atol=1e-6)

    # recovered subspace should closely match the true one
    from scipy.linalg import subspace_angles

    angles = subspace_angles(basis, u)
    assert np.max(angles) < 0.1


def test_class_centroid_pca_separates_class_means():
    rng = np.random.default_rng(1)
    d = 10
    centroids = {0: rng.standard_normal(d) * 5, 1: rng.standard_normal(d) * 5, 2: rng.standard_normal(d) * 5}
    labels = np.repeat([0, 1, 2], 20)
    features = np.stack([centroids[label] + 0.01 * rng.standard_normal(d) for label in labels])

    u = class_centroid_pca(features, labels, rank=2)
    assert u.shape == (d, 2)
    assert np.allclose(u.T @ u, np.eye(2), atol=1e-6)


def test_probe_nullspace_removal_returns_orthonormal_basis():
    rng = np.random.default_rng(2)
    d, n = 15, 200
    features = rng.standard_normal((n, d))
    labels = (features[:, 0] > 0).astype(int)  # linearly separable on dim 0

    u = probe_nullspace_removal(features, labels, rank=1)
    assert u.shape[0] == d
    assert np.allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-6)


def test_low_rank_svd_factorization_shape_and_orthonormality():
    rng = np.random.default_rng(3)
    features = rng.standard_normal((100, 20))
    u = low_rank_svd_factorization(features, rank=4)
    assert u.shape == (20, 4)
    assert np.allclose(u.T @ u, np.eye(4), atol=1e-6)


def test_random_subspace_baseline_orthonormal_and_reproducible():
    u1 = random_subspace_baseline(20, 4, seed=0)
    u2 = random_subspace_baseline(20, 4, seed=0)
    u3 = random_subspace_baseline(20, 4, seed=1)
    assert np.allclose(u1, u2)
    assert not np.allclose(u1, u3)
    assert np.allclose(u1.T @ u1, np.eye(4), atol=1e-6)
