import numpy as np
import pandas as pd

from isplit.eval.probe_vs_intervention import correlate_metrics, ranking_disagreement


def _fake_audit_df(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for encoder in ["enc_a", "enc_b"]:
        for layer in range(6):
            decodability = rng.uniform(0.5, 1.0)
            css = decodability + rng.normal(0, 0.05)  # correlated but noisy
            iei = rng.uniform(0, 1)
            rows.append(
                {"encoder": encoder, "layer": layer, "factor": "speaker", "decodability": decodability, "css": css, "iei": iei}
            )
    return pd.DataFrame(rows)


def test_correlate_metrics_detects_strong_correlation():
    df = _fake_audit_df()
    result = correlate_metrics(df)
    row = result[(result.metric_a == "decodability") & (result.metric_b == "css")].iloc[0]
    assert row["spearman_rho"] > 0.5


def test_correlate_metrics_handles_insufficient_data():
    df = pd.DataFrame({"decodability": [0.9], "css": [0.8], "iei": [0.5]})
    result = correlate_metrics(df)
    assert result["n"].max() <= 1


def test_ranking_disagreement_flags_gaps():
    df = pd.DataFrame(
        [
            {"encoder": "enc_a", "layer": 0, "factor": "speaker", "decodability": 0.9, "css": 0.2},
            {"encoder": "enc_a", "layer": 1, "factor": "speaker", "decodability": 0.2, "css": 0.9},
            {"encoder": "enc_a", "layer": 2, "factor": "speaker", "decodability": 0.5, "css": 0.5},
        ]
    )
    result = ranking_disagreement(df, "decodability", "css")
    assert len(result) == 3
    assert result.iloc[0]["rank_gap"] >= result.iloc[-1]["rank_gap"]


def test_ranking_disagreement_empty_when_insufficient_rows():
    df = pd.DataFrame({"encoder": ["enc_a"], "layer": [0], "factor": ["speaker"], "decodability": [0.9], "css": [0.5]})
    result = ranking_disagreement(df, "decodability", "css")
    assert result.empty
