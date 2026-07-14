"""Materialize VCTK-0.92 from the Hugging Face parquet mirror into the on-disk
layout `build_vctk_manifest` expects (`<spk>/<utt_id>_mic1.flac` + `<utt_id>.txt`).

Why this exists: `scripts/download_data.py` pulls VCTK from the official
Edinburgh DataShare host, which served this machine at ~50 kB/s (~70h for the
11.7GB zip). `sanchit-gandhi/vctk` mirrors the identical corpus (same speakers,
same `text_id`s, same mic1/mic2 flac payloads) as parquet shards that pull at
tens of MB/s. Audio bytes are written through verbatim -- no re-encoding.

Two trees are produced:

  <out>/vctk         every prompt of every speaker -- the real corpus, for --scale full
  <out>/vctk_common  only utterances whose text is read by >= --min-speakers-per-text
                     speakers (default 50, which selects the ~24 elicitation
                     prompts every VCTK speaker reads)

The second tree exists because of how VCTK is built. Only the elicitation
prompts are the same sentence for every speaker; past those, each speaker reads
their own newspaper sentences -- reusing the same `text_id` for *different*
text, while occasionally sharing a sentence with a handful of other speakers.
Measured on the corpus: ~24 texts are read by 94-108 speakers, then a long tail
of 8k texts shared by between 2 and 33.

Same-text/different-speaker pairs can only be drawn from texts two sampled
speakers both happen to have. Pilot scale keeps 15 random utterances of each
speaker's ~360, so a tail text (shared by a handful of speakers, each with a
~4% chance of having sampled it) essentially never survives in two speakers at
once -- the speaker factor would be left with a handful of pairs instead of the
400 it asks for. Restricting to the universal prompts means every speaker keeps
~24 candidates, 15 of which survive the cap, so same-text pairs are dense.
`configs/pilot.yaml` points `data.vctk_dir` here. Entries are hardlinks into
<out>/vctk -- no second copy of the audio.

Usage: uv run python scripts/materialize_vctk_hf.py
"""

import collections
import io
from pathlib import Path

import click
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from isplit.utils.logging import get_logger

logger = get_logger(__name__)

REPO_ID = "sanchit-gandhi/vctk"


def _text_key(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def materialize_full(full_dir: Path, max_shards: int | None) -> int:
    """Write every mic1 utterance of the corpus as <spk>/<utt_id>_mic1.flac + <utt_id>.txt."""
    if (full_dir / ".done").exists():
        logger.info("full corpus already materialized at %s", full_dir)
        return sum(1 for _ in full_dir.rglob("*.flac"))

    shards = sorted(f for f in list_repo_files(REPO_ID, repo_type="dataset") if f.endswith(".parquet"))
    if max_shards:
        shards = shards[:max_shards]
    logger.info("Materializing VCTK from %s (%d shards)", REPO_ID, len(shards))

    n_audio = 0
    for i, shard in enumerate(shards, 1):
        path = hf_hub_download(REPO_ID, shard, repo_type="dataset")
        table = pq.read_table(path, columns=["speaker_id", "text", "audio"])
        for row in table.to_pylist():
            spk, text, audio = row["speaker_id"], row["text"], row["audio"]
            name = Path(audio["path"]).stem  # e.g. p225_001_mic1
            if not name.endswith("_mic1"):  # manifest prefers mic1; skip the mic2 duplicate
                continue
            utt_id = name[: -len("_mic1")]
            spk_dir = full_dir / spk
            spk_dir.mkdir(parents=True, exist_ok=True)

            flac_path = spk_dir / f"{name}.flac"
            if not flac_path.exists():
                flac_path.write_bytes(audio["bytes"])
            txt_path = spk_dir / f"{utt_id}.txt"
            if not txt_path.exists():
                txt_path.write_text(text.strip() + "\n", encoding="utf-8")
            n_audio += 1
        logger.info("shard %d/%d: %d mic1 utterances so far", i, len(shards), n_audio)

    (full_dir / ".done").write_text(f"materialized {n_audio} utterances from {REPO_ID}\n")
    return n_audio


def materialize_common(full_dir: Path, common_dir: Path, min_speakers_per_text: int) -> None:
    """Hardlink the subset of `full_dir` whose text is read by many speakers."""
    speakers_of_text: dict[str, set[str]] = collections.defaultdict(set)
    utts: list[tuple[str, str, str]] = []  # (speaker_id, utt_id, text_key)
    for txt_path in full_dir.rglob("*.txt"):
        tk = _text_key(txt_path.read_text(encoding="utf-8"))
        spk = txt_path.parent.name
        speakers_of_text[tk].add(spk)
        utts.append((spk, txt_path.stem, tk))

    shared = {tk for tk, spks in speakers_of_text.items() if len(spks) >= min_speakers_per_text}
    n_linked = 0
    for spk, utt_id, tk in utts:
        if tk not in shared:
            continue
        dst_dir = common_dir / spk
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in (full_dir / spk / f"{utt_id}_mic1.flac", full_dir / spk / f"{utt_id}.txt"):
            dst = dst_dir / src.name
            if not dst.exists():
                dst.hardlink_to(src)
        n_linked += 1

    (common_dir / ".done").write_text(f"{n_linked} utterances over {len(shared)} shared texts\n")
    logger.info(
        "vctk_common/ : %d utterances over %d texts read by >=%d speakers (%d speakers, ~%.0f utts each)",
        n_linked, len(shared), min_speakers_per_text,
        len({s for s, _, tk in utts if tk in shared}),
        n_linked / max(1, len({s for s, _, tk in utts if tk in shared})),
    )


@click.command()
@click.option("--out-dir", default="data/raw", help="parent dir for vctk/ and vctk_common/")
@click.option("--max-shards", default=None, type=int, help="limit shards (debug); default all")
@click.option(
    "--min-speakers-per-text",
    default=50,
    help="a text must be read by this many speakers to enter vctk_common (see module docstring)",
)
def main(out_dir: str, max_shards: int | None, min_speakers_per_text: int) -> None:
    out = Path(out_dir)
    full_dir, common_dir = out / "vctk", out / "vctk_common"
    full_dir.mkdir(parents=True, exist_ok=True)

    n_audio = materialize_full(full_dir, max_shards)
    logger.info("vctk/        : %d utterances, %d speakers", n_audio, len(list(full_dir.glob("p*"))))
    materialize_common(full_dir, common_dir, min_speakers_per_text)


if __name__ == "__main__":
    main()
