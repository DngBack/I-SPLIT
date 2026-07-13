import pandas as pd

from isplit.data.requests import collect_feature_requests


def test_collect_feature_requests_content_and_speaker():
    pair_sets = {
        "speaker": pd.DataFrame(
            [{"a_utt_id": "p1_001", "b_utt_id": "p2_001", "split": "train"}]
        ),
        "content": pd.DataFrame(
            [{"a_utt_id": "p1_001", "b_utt_id": "p1_002", "split": "train"}]
        ),
    }
    reqs = collect_feature_requests(pair_sets)
    keys = set(zip(reqs["utt_id"], reqs["condition"], strict=True))
    assert ("p1_001", None) in keys
    assert ("p2_001", None) in keys
    assert ("p1_002", None) in keys


def test_collect_feature_requests_deduplicates():
    pair_sets = {
        "speaker": pd.DataFrame(
            [
                {"a_utt_id": "p1_001", "b_utt_id": "p2_001", "split": "train"},
                {"a_utt_id": "p1_001", "b_utt_id": "p3_001", "split": "train"},
            ]
        )
    }
    reqs = collect_feature_requests(pair_sets)
    # p1_001 requested twice (once per pair) but should be deduplicated
    assert len(reqs[(reqs.utt_id == "p1_001") & (reqs.condition.isna())]) == 1


def test_collect_feature_requests_environment_conditions():
    pair_sets = {
        "environment": pd.DataFrame(
            [
                {
                    "base_utt_id": "p1_001",
                    "a_noise_id": "clean",
                    "a_snr_db": None,
                    "b_noise_id": "n1",
                    "b_snr_db": 10.0,
                    "split": "train",
                }
            ]
        )
    }
    reqs = collect_feature_requests(pair_sets)
    conditions = set(reqs["condition"])
    assert "clean" in conditions
    assert "noise=n1_snr=10.0" in conditions


def test_collect_feature_requests_channel_conditions():
    pair_sets = {
        "channel": pd.DataFrame(
            [{"base_utt_id": "p1_001", "a_channel": "clean", "b_channel": "telephone", "split": "held_out"}]
        )
    }
    reqs = collect_feature_requests(pair_sets)
    conditions = set(reqs["condition"])
    assert "channel=clean" in conditions
    assert "channel=telephone" in conditions
    assert (reqs["split"] == "held_out").all()


def test_collect_feature_requests_empty_input():
    reqs = collect_feature_requests({})
    assert reqs.empty
