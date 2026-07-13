import numpy as np

from isplit.theory.validate_propositions import (
    prop1_eigenspace_identifiability,
    prop1_noise_sensitivity,
    prop2_oblique_vs_orthogonal,
)


def test_prop1_recovery_error_shrinks_with_sample_size():
    df = prop1_eigenspace_identifiability(
        ambient_dim=32, factor_rank=3, n_values=(10, 50, 500, 5000), seed=0
    )
    angles = df.sort_values("n")["mean_angle_deg"].to_numpy()
    # not strictly monotonic at small n due to noise, but the largest-N estimate
    # must be much better than the smallest-N estimate
    assert angles[-1] < angles[0] / 2


def test_prop1_recovery_error_grows_with_noise():
    df = prop1_noise_sensitivity(
        ambient_dim=32, factor_rank=3, n=500, noise_std_values=(0.01, 0.1, 1.0, 5.0), seed=0
    )
    angles = df.sort_values("noise_std")["mean_angle_deg"].to_numpy()
    assert angles[-1] > angles[0]


def test_prop2_condition_number_increases_as_angle_shrinks():
    df = prop2_oblique_vs_orthogonal(
        ambient_dim=32, content_rank=2, nuisance_rank=2, angles_deg=(90, 30, 5, 0.5), seed=0
    )
    cond = df.sort_values("principal_angle_deg", ascending=False)["condition_number_basis"].to_numpy()
    assert np.all(np.diff(cond) > 0)  # condition number strictly increases as angle shrinks


def test_prop2_oblique_content_error_finite_and_bounded_at_moderate_angle():
    df = prop2_oblique_vs_orthogonal(
        ambient_dim=32,
        content_rank=2,
        nuisance_rank=2,
        angles_deg=(60,),
        tau=1e-2,
        noise_std=0.1,
        seed=0,
    )
    row = df.iloc[0]
    assert np.isfinite(row["oblique_content_error"])
    assert row["oblique_content_error"] < 1.0


def test_prop2_orthogonal_baseline_degrades_as_angle_shrinks():
    df = prop2_oblique_vs_orthogonal(
        ambient_dim=32, content_rank=2, nuisance_rank=2, angles_deg=(90, 5), noise_std=0.1, seed=0
    )
    df = df.sort_values("principal_angle_deg", ascending=False)
    # orthogonal removal assumes the nuisance direction it removes IS the nuisance
    # subspace; at a small angle it also eats into content it shouldn't -- error
    # should not improve when the angle shrinks (it may already be poor at 90deg,
    # but it should not get *better* as the geometry gets harder).
    assert df.iloc[1]["orthogonal_content_error"] >= df.iloc[0]["orthogonal_content_error"] - 1e-6
