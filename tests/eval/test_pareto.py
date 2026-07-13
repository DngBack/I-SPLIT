import pandas as pd

from isplit.eval.pareto import pareto_frontier_area


def test_pareto_frontier_area_perfect_frontier_is_one():
    df = pd.DataFrame({"nuisance_erasure": [0.0, 1.0], "content_preservation": [1.0, 1.0]})
    assert pareto_frontier_area(df) == 1.0


def test_pareto_frontier_area_worst_frontier_is_zero():
    df = pd.DataFrame({"nuisance_erasure": [0.0, 1.0], "content_preservation": [0.0, 0.0]})
    assert pareto_frontier_area(df) == 0.0


def test_pareto_frontier_area_higher_curve_scores_higher():
    good = pd.DataFrame({"nuisance_erasure": [0.0, 0.5, 1.0], "content_preservation": [1.0, 0.9, 0.8]})
    bad = pd.DataFrame({"nuisance_erasure": [0.0, 0.5, 1.0], "content_preservation": [1.0, 0.3, 0.1]})
    assert pareto_frontier_area(good) > pareto_frontier_area(bad)
