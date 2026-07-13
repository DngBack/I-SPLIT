import pandas as pd
import pytest

from isplit.data.pairs import (
    assign_split_by_speaker,
    build_all_pairs,
    make_channel_pairs,
    make_content_pairs,
    make_environment_pairs,
    make_speaker_pairs,
    split_speakers,
)


def _fake_vctk_manifest() -> pd.DataFrame:
    # 4 speakers x 3 prompts, all speakers read all prompts (like real VCTK)
    speakers = ["p001", "p002", "p003", "p004"]
    prompts = ["001", "002", "003"]
    rows = []
    for sid in speakers:
        for pid in prompts:
            rows.append(
                {
                    "utt_id": f"{sid}_{pid}",
                    "speaker_id": sid,
                    "prompt_id": pid,
                    "text": f"sentence {pid}",
                    "wav_path": f"/fake/{sid}_{pid}.flac",
                }
            )
    return pd.DataFrame(rows)


def _fake_noise_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"noise_id": "n1", "category": "free-sound", "wav_path": "/fake/n1.wav"},
            {"noise_id": "n2", "category": "free-sound", "wav_path": "/fake/n2.wav"},
        ]
    )


def test_split_speakers_disjoint_and_covers_all():
    speakers = [f"p{i:03d}" for i in range(20)]
    train, held_out = split_speakers(speakers, held_out_fraction=0.2, seed=0)
    assert train.isdisjoint(held_out)
    assert train | held_out == set(speakers)
    assert len(held_out) == 4


def test_speaker_pairs_isolate_speaker_factor_only():
    manifest = _fake_vctk_manifest()
    pairs = make_speaker_pairs(manifest, n_pairs=20, seed=0)
    assert len(pairs) > 0
    for _, row in pairs.iterrows():
        assert row["a_speaker_id"] != row["b_speaker_id"]
        # same prompt_id (text) by construction -- factor is isolated to speaker
        a_prompt = manifest.loc[manifest.utt_id == row["a_utt_id"], "prompt_id"].iloc[0]
        b_prompt = manifest.loc[manifest.utt_id == row["b_utt_id"], "prompt_id"].iloc[0]
        assert a_prompt == b_prompt == row["prompt_id"]


def test_content_pairs_isolate_content_factor_only():
    manifest = _fake_vctk_manifest()
    pairs = make_content_pairs(manifest, n_pairs=20, seed=0)
    assert len(pairs) > 0
    for _, row in pairs.iterrows():
        assert row["a_prompt_id"] != row["b_prompt_id"]
        a_speaker = manifest.loc[manifest.utt_id == row["a_utt_id"], "speaker_id"].iloc[0]
        b_speaker = manifest.loc[manifest.utt_id == row["b_utt_id"], "speaker_id"].iloc[0]
        assert a_speaker == b_speaker == row["speaker_id"]


def test_environment_pairs_share_identical_base_utterance():
    manifest = _fake_vctk_manifest()
    noise = _fake_noise_manifest()
    pairs = make_environment_pairs(manifest, noise, snr_levels_db=[0.0, 10.0], n_pairs=20, seed=0)
    assert len(pairs) > 0
    for _, row in pairs.iterrows():
        # exactly one underlying utterance -- only noise/SNR differs
        cond_a = (row["a_noise_id"], row["a_snr_db"])
        cond_b = (row["b_noise_id"], row["b_snr_db"])
        assert cond_a != cond_b
        assert row["base_utt_id"] in manifest["utt_id"].to_numpy()


def test_channel_pairs_share_identical_base_utterance():
    manifest = _fake_vctk_manifest()
    pairs = make_channel_pairs(manifest, n_pairs=20, seed=0)
    assert len(pairs) > 0
    for _, row in pairs.iterrows():
        assert row["a_channel"] != row["b_channel"]
        assert row["base_utt_id"] in manifest["utt_id"].to_numpy()


def test_assign_split_drops_mixed_speaker_pairs():
    df = pd.DataFrame(
        [
            {"a_speaker_id": "p1", "b_speaker_id": "p2"},  # both train
            {"a_speaker_id": "p1", "b_speaker_id": "p3"},  # mixed -> dropped
            {"a_speaker_id": "p3", "b_speaker_id": "p3"},  # both held-out
        ]
    )
    train_speakers, held_out_speakers = {"p1", "p2"}, {"p3"}
    out = assign_split_by_speaker(df, ["a_speaker_id", "b_speaker_id"], train_speakers, held_out_speakers)
    assert len(out) == 2
    assert set(out["split"]) == {"train", "held_out"}


def test_build_all_pairs_never_leaks_held_out_speakers_into_train():
    manifest = _fake_vctk_manifest()
    noise = _fake_noise_manifest()
    pair_sets = build_all_pairs(
        manifest,
        noise,
        snr_levels_db=[0.0, 10.0],
        n_pairs_per_factor={"speaker": 20, "content": 20, "environment": 20, "channel": 20},
        held_out_fraction=0.25,
        seed=0,
    )

    train_speakers, held_out_speakers = split_speakers(manifest["speaker_id"], 0.25, seed=0)
    assert train_speakers.isdisjoint(held_out_speakers)

    for factor, df in pair_sets.items():
        if df.empty:
            continue
        assert set(df["split"].unique()) <= {"train", "held_out"}
        speaker_cols = [c for c in df.columns if "speaker_id" in c]
        for _, row in df.iterrows():
            speakers_in_row = {row[c] for c in speaker_cols}
            if row["split"] == "train":
                assert speakers_in_row <= train_speakers, factor
            else:
                assert speakers_in_row <= held_out_speakers, factor


def test_build_all_pairs_uses_consistent_split_across_factors():
    # A speaker held out for one factor must be held out for every factor,
    # since generalization claims compare across factors on the same split.
    manifest = _fake_vctk_manifest()
    noise = _fake_noise_manifest()
    pair_sets = build_all_pairs(
        manifest,
        noise,
        snr_levels_db=[0.0],
        n_pairs_per_factor={"speaker": 10, "content": 10, "environment": 10, "channel": 10},
        held_out_fraction=0.25,
        seed=0,
    )
    held_out_speakers_seen = set()
    train_speakers_seen = set()
    for df in pair_sets.values():
        if df.empty:
            continue
        speaker_cols = [c for c in df.columns if "speaker_id" in c]
        for _, row in df.iterrows():
            speakers_in_row = {row[c] for c in speaker_cols}
            if row["split"] == "held_out":
                held_out_speakers_seen |= speakers_in_row
            else:
                train_speakers_seen |= speakers_in_row
    assert held_out_speakers_seen.isdisjoint(train_speakers_seen)


def test_make_speaker_pairs_empty_when_no_shared_prompts():
    manifest = pd.DataFrame(
        [
            {"utt_id": "p1_001", "speaker_id": "p1", "prompt_id": "001", "text": "a", "wav_path": "x"},
            {"utt_id": "p2_002", "speaker_id": "p2", "prompt_id": "002", "text": "b", "wav_path": "y"},
        ]
    )
    pairs = make_speaker_pairs(manifest, n_pairs=10, seed=0)
    assert pairs.empty
