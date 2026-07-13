from pathlib import Path

import pandas as pd

from isplit.data.manifest import (
    apply_file_cap,
    apply_pilot_caps,
    build_librispeech_manifest,
    build_musan_noise_manifest,
    build_vctk_manifest,
)


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_vctk_manifest_prefers_mic1_and_parses_ids(tmp_path):
    vctk_dir = tmp_path / "vctk"
    _write(vctk_dir / "txt" / "p225" / "p225_001.txt", "Please call Stella.")
    _write(vctk_dir / "wav48_silence_trimmed" / "p225" / "p225_001_mic1.flac")
    _write(vctk_dir / "wav48_silence_trimmed" / "p225" / "p225_001_mic2.flac")
    # a non-utterance txt file that must NOT be picked up (e.g. speaker-info.txt)
    _write(vctk_dir / "speaker-info.txt", "ID AGE GENDER ACCENTS REGION")

    manifest = build_vctk_manifest(vctk_dir)

    assert len(manifest) == 1
    row = manifest.iloc[0]
    assert row["speaker_id"] == "p225"
    assert row["prompt_id"] == "001"
    assert row["text"] == "Please call Stella."
    assert row["wav_path"].endswith("_mic1.flac")


def test_build_vctk_manifest_skips_utterance_missing_audio(tmp_path):
    vctk_dir = tmp_path / "vctk"
    _write(vctk_dir / "txt" / "p225" / "p225_001.txt", "text only, no audio")
    manifest = build_vctk_manifest(vctk_dir)
    assert manifest.empty


def test_build_musan_noise_manifest_finds_all_wavs(tmp_path):
    musan_dir = tmp_path / "musan"
    _write(musan_dir / "noise" / "free-sound" / "noise-free-sound-0000.wav")
    _write(musan_dir / "noise" / "sound-bible" / "noise-sound-bible-0000.wav")

    manifest = build_musan_noise_manifest(musan_dir)

    assert len(manifest) == 2
    assert set(manifest["category"]) == {"free-sound", "sound-bible"}


def test_build_librispeech_manifest_parses_transcripts(tmp_path):
    ls_dir = tmp_path / "librispeech"
    chapter_dir = ls_dir / "dev-clean" / "1272" / "128104"
    _write(
        chapter_dir / "1272-128104.trans.txt",
        "1272-128104-0000 MISTER QUILTER IS THE APOSTLE\n"
        "1272-128104-0001 NOR IS MISTER QUILTER'S MANNER LESS\n",
    )
    _write(chapter_dir / "1272-128104-0000.flac")
    # deliberately omit audio for the second utterance -- must be skipped

    manifest = build_librispeech_manifest(ls_dir)

    assert len(manifest) == 1
    assert manifest.iloc[0]["speaker_id"] == "1272"
    assert manifest.iloc[0]["text"] == "MISTER QUILTER IS THE APOSTLE"


def test_apply_pilot_caps_respects_max_speakers_and_utterances():
    rows = []
    for sid in [f"p{i:03d}" for i in range(10)]:
        for j in range(5):
            rows.append({"utt_id": f"{sid}_{j}", "speaker_id": sid})
    df = pd.DataFrame(rows)

    capped = apply_pilot_caps(df, max_speakers=3, max_utterances_per_speaker=2, seed=0)

    assert capped["speaker_id"].nunique() == 3
    assert (capped.groupby("speaker_id").size() <= 2).all()


def test_apply_pilot_caps_noop_when_no_caps():
    df = pd.DataFrame({"utt_id": ["a", "b"], "speaker_id": ["p1", "p2"]})
    capped = apply_pilot_caps(df, max_speakers=None, max_utterances_per_speaker=None, seed=0)
    assert len(capped) == 2


def test_apply_file_cap_limits_row_count():
    df = pd.DataFrame({"noise_id": [f"n{i}" for i in range(50)]})
    capped = apply_file_cap(df, max_files=10, seed=0)
    assert len(capped) == 10
