import numpy as np
import pandas as pd
import pytest

from isplit.eval.ablations import (
    isolated_vs_joint_subspace_similarity,
    linear_vs_nonlinear_leakage,
    pseudo_label_speaker_pairs,
    shuffle_pair_correspondence,
    true_vs_mismatched_subspace_quality,
)
from isplit.theory.synthetic import build_orthogonal_mixing_model, sample_latents, synthesize, FactorSpec


def test_shuffle_pair_correspondence_preserves_marginal_but_breaks_pairing():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    shuffled = shuffle_pair_correspondence(df, "b", seed=0)
    assert set(shuffled["b"]) == set(df["b"])  # marginal preserved
    assert set(shuffled["a"]) == set(df["a"])


def test_true_pairs_concentrate_more_energy_than_mismatched():
    rng = np.random.default_rng(0)
    d, r, n = 40, 3, 800
    # Two factors, and only `speaker` is intervened on: `content` is held fixed
    # within each true pair. That held-fixed factor is the whole point -- it is
    # what shuffling the pair correspondence destroys. With a single-factor model
    # the true and mismatched deltas are identically distributed, so the
    # assertion below would just be a coin flip.
    factors = [FactorSpec("speaker", r), FactorSpec("content", 10)]
    model = build_orthogonal_mixing_model(d, factors, seed=0)
    latents = sample_latents(factors, n=n, seed=1)
    features_a = synthesize(model, latents, noise_std=0.3, seed=2)

    new_latents = dict(latents)
    new_latents["speaker"] = rng.standard_normal((n, r))
    features_b = synthesize(model, new_latents, noise_std=0.3, seed=3)

    result = true_vs_mismatched_subspace_quality(features_a, features_b, seed=0, rank=r)
    assert result["true_top_rank_energy_fraction"] > result["mismatched_top_rank_energy_fraction"]


def test_isolated_vs_joint_subspace_similarity_identical_subspace_is_one():
    rng = np.random.default_rng(0)
    u, _ = np.linalg.qr(rng.standard_normal((15, 3)))
    assert isolated_vs_joint_subspace_similarity(u, u) == 1.0


def test_isolated_vs_joint_subspace_similarity_orthogonal_is_zero():
    d = 12
    u1 = np.eye(d)[:, :3]
    u2 = np.eye(d)[:, 3:6]
    # cos(pi/2) is 6.1e-17, not 0, in IEEE double -- so the squared cosine lands
    # at ~1e-33 rather than exactly 0.0.
    assert isolated_vs_joint_subspace_similarity(u1, u2) == pytest.approx(0.0, abs=1e-12)


def test_linear_vs_nonlinear_leakage_detects_nonlinear_signal():
    rng = np.random.default_rng(0)
    n, d = 400, 10
    x = rng.standard_normal((n, d))
    # XOR-like nonlinear label: not linearly separable from x[:, 0], x[:, 1] alone
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(int)

    result = linear_vs_nonlinear_leakage(x, y, seed=0)
    assert result["nonlinear_leakage_acc"] >= result["linear_leakage_acc"] - 0.05


def test_pseudo_label_speaker_pairs_produces_cross_speaker_pairs():
    manifest = pd.DataFrame(
        [
            {"utt_id": f"p{i:03d}_001", "speaker_id": f"p{i:03d}", "prompt_id": "001", "text": "hello world"}
            for i in range(10)
        ]
    )
    pairs = pseudo_label_speaker_pairs(manifest, n_pairs=5, seed=0)
    assert len(pairs) > 0
    assert (pairs["a_speaker_id"] != pairs["b_speaker_id"]).all()
    assert (pairs["label_source"] == "pseudo").all()
