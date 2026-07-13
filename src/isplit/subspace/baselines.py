"""Baseline subspace-identification methods I-SPLIT is compared against:
plain PCA, class-centroid PCA, probe-nullspace (concept-erasure) removal, and
low-rank SVD factorization (LinearVC-style). All return an orthonormal basis
U (D, rank) so they're drop-in comparable with `subspace.intervention_covariance`
estimates and `subspace.projection` reconstruction/removal utilities.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from isplit.utils.linalg import orthonormalize


def plain_pca(features: np.ndarray, rank: int) -> np.ndarray:
    """Top-`rank` principal directions of `features` (D, rank), no factor labels used."""
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:rank].T


def class_centroid_pca(features: np.ndarray, labels: np.ndarray, rank: int) -> np.ndarray:
    """PCA over per-class centroids rather than raw examples -- isolates the
    directions along which class means vary, following the speaker/phonetic
    orthogonality-analysis literature this method is compared against.
    """
    classes = np.unique(labels)
    centroids = np.stack([features[labels == c].mean(axis=0) for c in classes])
    return plain_pca(centroids, rank=min(rank, centroids.shape[0] - 1))


def probe_nullspace_removal(features: np.ndarray, labels: np.ndarray, rank: int) -> np.ndarray:
    """Concept-erasure-style baseline: fit a linear probe for `labels`, take the
    directions most predictive of the label (probe weight vectors, one-vs-rest)
    as the "factor subspace" to remove. A simplified INLP-style single-pass
    variant (not iterative nullspace projection), which is standard practice
    for a lightweight baseline in this comparison.
    """
    clf = LogisticRegression(max_iter=1000)
    clf.fit(features, labels)
    weights = np.atleast_2d(clf.coef_)  # (n_classes_or_1, D)
    rank = min(rank, weights.shape[0], features.shape[1])
    return orthonormalize(weights[:rank].T)


def low_rank_svd_factorization(features: np.ndarray, rank: int) -> np.ndarray:
    """LinearVC-style low-rank linear factorization: top-`rank` right-singular
    vectors of the (uncentered) feature matrix.
    """
    _, _, vt = np.linalg.svd(features, full_matrices=False)
    return vt[:rank].T


def random_subspace_baseline(ambient_dim: int, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return orthonormalize(rng.standard_normal((ambient_dim, rank)))
