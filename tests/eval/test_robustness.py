import numpy as np
import pandas as pd

from isplit.eval.robustness import held_out_vs_train_css_gap, joint_unseen_combination_pairs


def test_held_out_vs_train_css_gap_basic():
    assert held_out_vs_train_css_gap(0.8, 0.6) == pytest_approx(0.2)
    assert held_out_vs_train_css_gap(0.5, 0.5) == pytest_approx(0.0)


def test_held_out_vs_train_css_gap_nan_propagates():
    assert np.isnan(held_out_vs_train_css_gap(float("nan"), 0.5))
    assert np.isnan(held_out_vs_train_css_gap(0.5, float("nan")))


def test_joint_unseen_combination_pairs_filters_seen_combos():
    pairs = pd.DataFrame(
        [
            {"noise_id": "n1", "channel": "clean", "split": "train"},
            {"noise_id": "n2", "channel": "clean", "split": "train"},
            {"noise_id": "n1", "channel": "clean", "split": "held_out"},  # combo seen in train -> excluded
            {"noise_id": "n3", "channel": "telephone", "split": "held_out"},  # unseen combo -> kept
        ]
    )
    result = joint_unseen_combination_pairs(pairs, ["noise_id", "channel"])
    assert len(result) == 1
    assert result.iloc[0]["noise_id"] == "n3"


def test_joint_unseen_combination_pairs_empty_input():
    empty = pd.DataFrame(columns=["noise_id", "channel", "split"])
    result = joint_unseen_combination_pairs(empty, ["noise_id", "channel"])
    assert result.empty


def pytest_approx(value):
    import pytest

    return pytest.approx(value, abs=1e-9)
