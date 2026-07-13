import numpy as np
import pytest

from isplit.causal.entanglement import irreducible_entanglement_index, pairwise_entanglement_matrix
from isplit.theory.synthetic import make_two_subspaces


def test_iei_identical_subspace_is_one():
    rng = np.random.default_rng(0)
    u, _ = np.linalg.qr(rng.standard_normal((10, 3)))
    assert irreducible_entanglement_index(u, u) == pytest.approx(1.0, abs=1e-6)


def test_iei_orthogonal_subspace_is_zero():
    d = 10
    u1 = np.eye(d)[:, :3]
    u2 = np.eye(d)[:, 3:6]
    assert irreducible_entanglement_index(u1, u2) == pytest.approx(0.0, abs=1e-6)


def test_iei_matches_known_controlled_angle():
    u1, u2 = make_two_subspaces(ambient_dim=20, dim1=1, dim2=1, principal_angle_deg=30, seed=0)
    iei = irreducible_entanglement_index(u1, u2)
    expected = np.cos(np.deg2rad(30)) ** 2  # cos^2(30deg) = 0.75
    assert iei == pytest.approx(expected, abs=1e-6)


def test_pairwise_entanglement_matrix_diagonal_is_one_and_symmetric():
    rng = np.random.default_rng(1)
    subspaces = {
        "content": np.linalg.qr(rng.standard_normal((15, 3)))[0],
        "speaker": np.linalg.qr(rng.standard_normal((15, 3)))[0],
    }
    mat = pairwise_entanglement_matrix(subspaces)
    assert mat.loc["content", "content"] == 1.0
    assert mat.loc["speaker", "speaker"] == 1.0
    assert mat.loc["content", "speaker"] == pytest.approx(mat.loc["speaker", "content"], abs=1e-10)
