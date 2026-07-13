"""Build tabular manifests (speaker/text/path metadata) from the raw VCTK,
MUSAN, and LibriSpeech directory trees. File discovery is done by recursive
pattern search rather than hardcoded subfolder names, since corpus release
layouts (e.g. VCTK's `wav48_silence_trimmed/` vs older `wav48/`) have shifted
across versions and mirrors.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

_VCTK_UTT_ID_RE = re.compile(r"^p\d+_\d+$")


def build_vctk_manifest(vctk_dir: str | Path) -> pd.DataFrame:
    """VCTK: same fixed prompts read by ~110 speakers -- `prompt_id` is what
    makes exact same-text/different-speaker pairs possible (see data/pairs.py).
    Prefers mic1 recordings when both mic1/mic2 exist for an utterance.
    """
    vctk_dir = Path(vctk_dir)

    text_by_utt: dict[str, Path] = {}
    for p in vctk_dir.rglob("*.txt"):
        if _VCTK_UTT_ID_RE.match(p.stem):
            text_by_utt[p.stem] = p

    audio_by_utt: dict[str, Path] = {}
    for p in vctk_dir.rglob("*.flac"):
        stem = p.stem
        is_mic1 = stem.endswith("_mic1")
        for suffix in ("_mic1", "_mic2"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if not _VCTK_UTT_ID_RE.match(stem):
            continue
        if stem not in audio_by_utt or is_mic1:
            audio_by_utt[stem] = p

    rows = []
    for utt_id, txt_path in text_by_utt.items():
        audio_path = audio_by_utt.get(utt_id)
        if audio_path is None:
            continue
        speaker_id, prompt_id = utt_id.split("_", 1)
        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        rows.append(
            {
                "utt_id": utt_id,
                "speaker_id": speaker_id,
                "prompt_id": prompt_id,
                "text": text,
                "wav_path": str(audio_path),
            }
        )
    return pd.DataFrame(rows, columns=["utt_id", "speaker_id", "prompt_id", "text", "wav_path"])


def build_musan_noise_manifest(musan_dir: str | Path) -> pd.DataFrame:
    musan_dir = Path(musan_dir)
    rows = [
        {"noise_id": p.stem, "category": p.parent.name, "wav_path": str(p)}
        for p in musan_dir.rglob("*.wav")
    ]
    return pd.DataFrame(rows, columns=["noise_id", "category", "wav_path"])


def build_librispeech_manifest(librispeech_dir: str | Path) -> pd.DataFrame:
    librispeech_dir = Path(librispeech_dir)
    rows = []
    for trans_path in librispeech_dir.rglob("*.trans.txt"):
        chapter_dir = trans_path.parent
        for line in trans_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            utt_id, _, text = line.partition(" ")
            audio_path = chapter_dir / f"{utt_id}.flac"
            if not audio_path.exists():
                continue
            speaker_id = utt_id.split("-")[0]
            rows.append(
                {
                    "utt_id": utt_id,
                    "speaker_id": speaker_id,
                    "text": text,
                    "wav_path": str(audio_path),
                }
            )
    return pd.DataFrame(rows, columns=["utt_id", "speaker_id", "text", "wav_path"])


def apply_pilot_caps(
    df: pd.DataFrame,
    max_speakers: int | None = None,
    max_utterances_per_speaker: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Subsample a manifest to pilot scale: cap number of speakers and, within
    each kept speaker, cap number of utterances. A no-op (besides index reset)
    when both caps are None, so the same function is safe to call under
    `full.yaml` too.
    """
    if df.empty:
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    speakers = np.sort(df["speaker_id"].unique())
    if max_speakers is not None and len(speakers) > max_speakers:
        speakers = rng.choice(speakers, size=max_speakers, replace=False)
        df = df[df["speaker_id"].isin(speakers)]

    if max_utterances_per_speaker is not None:
        df = df.groupby("speaker_id", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), max_utterances_per_speaker), random_state=seed),
            include_groups=True,
        )
    return df.reset_index(drop=True)


def apply_file_cap(df: pd.DataFrame, max_files: int | None = None, seed: int = 0) -> pd.DataFrame:
    if max_files is not None and len(df) > max_files:
        df = df.sample(n=max_files, random_state=seed)
    return df.reset_index(drop=True)
