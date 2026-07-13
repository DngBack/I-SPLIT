"""Dataset acquisition: VCTK (controlled factorial base), MUSAN noise
(environment intervention source), LibriSpeech dev-clean (natural held-out
validation). Downloads are resumable/idempotent -- re-running skips completed
steps via `.done` marker files, and extraction is verified before a `.done`
marker is written so a crash mid-extraction is auto-detected and retried.

Official hosts are used as the source-of-record (and citation) for VCTK and
LibriSpeech. For MUSAN, a verified raw-audio Hugging Face mirror
(FluidInference/musan) is used as the primary path since it's materially more
reliable for scripted/resumable download than scripting openslr.org directly;
the official openslr.org URL is kept as a fallback.
"""

import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from isplit.utils.logging import get_logger

logger = get_logger(__name__)

VCTK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
LIBRISPEECH_DEV_CLEAN_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
MUSAN_HF_MIRROR_REPO_ID = "FluidInference/musan"


def stream_download(url: str, dest_path: str | Path, chunk_size: int = 1 << 20) -> Path:
    """Streaming download with HTTP Range-based resume: if `dest_path` already
    has partial content, resumes from where it left off.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    existing_bytes = dest_path.stat().st_size if dest_path.exists() else 0

    headers = {"Range": f"bytes={existing_bytes}-"} if existing_bytes else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        if existing_bytes and response.status_code == 200:
            # server ignored our Range request -- restart from scratch
            existing_bytes = 0
        elif response.status_code not in (200, 206):
            response.raise_for_status()

        total = int(response.headers.get("content-length", 0)) + existing_bytes
        mode = "ab" if existing_bytes else "wb"
        with open(dest_path, mode) as f, tqdm(
            total=total, initial=existing_bytes, unit="B", unit_scale=True, desc=dest_path.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return dest_path


def _done_marker(dest_dir: Path) -> Path:
    return dest_dir / ".done"


def extract_zip_streaming(zip_path: Path, dest_dir: Path, expected_min_files: int = 1) -> Path:
    dest_dir = Path(dest_dir)
    marker = _done_marker(dest_dir)
    if marker.exists():
        logger.info("already extracted: %s", dest_dir)
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for member in tqdm(members, desc=f"extracting {zip_path.name}"):
            zf.extract(member, dest_dir)

    n_files = sum(1 for _ in dest_dir.rglob("*") if _.is_file())
    if n_files < expected_min_files:
        raise RuntimeError(
            f"extraction of {zip_path} looks incomplete: found {n_files} files, "
            f"expected >= {expected_min_files}"
        )
    marker.write_text(f"extracted {n_files} files\n")
    return dest_dir


def extract_tar_streaming(tar_path: Path, dest_dir: Path, expected_min_files: int = 1) -> Path:
    dest_dir = Path(dest_dir)
    marker = _done_marker(dest_dir)
    if marker.exists():
        logger.info("already extracted: %s", dest_dir)
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        members = tf.getmembers()
        for member in tqdm(members, desc=f"extracting {tar_path.name}"):
            tf.extract(member, dest_dir, filter="data")

    n_files = sum(1 for _ in dest_dir.rglob("*") if _.is_file())
    if n_files < expected_min_files:
        raise RuntimeError(
            f"extraction of {tar_path} looks incomplete: found {n_files} files, "
            f"expected >= {expected_min_files}"
        )
    marker.write_text(f"extracted {n_files} files\n")
    return dest_dir


def download_vctk(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    archive_path = raw_dir / "downloads" / "VCTK-Corpus-0.92.zip"
    dest_dir = raw_dir / "vctk"
    if _done_marker(dest_dir).exists():
        logger.info("VCTK already present at %s", dest_dir)
        return dest_dir
    stream_download(VCTK_URL, archive_path)
    return extract_zip_streaming(archive_path, dest_dir, expected_min_files=1000)


def download_musan_noise(raw_dir: str | Path) -> Path:
    """Download only the `noise/` subset of MUSAN (the subset I-SPLIT actually
    uses for environment interventions), preferring the verified HF raw-audio
    mirror and falling back to the official openslr.org full tarball.
    """
    raw_dir = Path(raw_dir)
    dest_dir = raw_dir / "musan"
    if _done_marker(dest_dir).exists():
        logger.info("MUSAN already present at %s", dest_dir)
        return dest_dir

    try:
        from huggingface_hub import snapshot_download

        snapshot_path = snapshot_download(
            repo_id=MUSAN_HF_MIRROR_REPO_ID,
            repo_type="dataset",
            allow_patterns=["noise/*"],
            local_dir=dest_dir,
        )
        n_files = sum(1 for _ in Path(snapshot_path).rglob("*.wav"))
        if n_files == 0:
            raise RuntimeError("HF mirror returned zero noise .wav files")
        _done_marker(dest_dir).write_text(f"extracted {n_files} noise files via HF mirror\n")
        return dest_dir
    except Exception as exc:  # noqa: BLE001 -- deliberate fallback on any mirror failure
        logger.warning("MUSAN HF mirror failed (%s); falling back to openslr.org", exc)

    archive_path = raw_dir / "downloads" / "musan.tar.gz"
    full_dir = raw_dir / "musan_full"
    stream_download(MUSAN_URL, archive_path)
    extract_tar_streaming(archive_path, full_dir, expected_min_files=100)

    dest_dir.mkdir(parents=True, exist_ok=True)
    noise_src = full_dir / "musan" / "noise"
    n_files = 0
    for wav_path in noise_src.rglob("*.wav"):
        target = dest_dir / "noise" / wav_path.relative_to(noise_src)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(wav_path.read_bytes())
        n_files += 1
    if n_files == 0:
        raise RuntimeError(f"no noise .wav files found under {noise_src}")
    _done_marker(dest_dir).write_text(f"extracted {n_files} noise files via openslr fallback\n")
    return dest_dir


def download_librispeech_dev_clean(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    archive_path = raw_dir / "downloads" / "dev-clean.tar.gz"
    dest_dir = raw_dir / "librispeech_dev_clean"
    if _done_marker(dest_dir).exists():
        logger.info("LibriSpeech dev-clean already present at %s", dest_dir)
        return dest_dir
    stream_download(LIBRISPEECH_DEV_CLEAN_URL, archive_path)
    return extract_tar_streaming(archive_path, dest_dir, expected_min_files=500)
