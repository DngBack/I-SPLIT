"""Download VCTK, MUSAN (noise subset), and LibriSpeech dev-clean into
data/raw/. Idempotent -- re-running skips anything already fully extracted.

Usage: uv run python scripts/download_data.py [--skip-vctk] [--skip-musan] [--skip-librispeech]
"""

import click

from isplit.data.acquisition import (
    download_librispeech_dev_clean,
    download_musan_noise,
    download_vctk,
)
from isplit.utils.logging import get_logger

logger = get_logger(__name__)


@click.command()
@click.option("--raw-dir", default="data/raw")
@click.option("--skip-vctk", is_flag=True)
@click.option("--skip-musan", is_flag=True)
@click.option("--skip-librispeech", is_flag=True)
def main(raw_dir: str, skip_vctk: bool, skip_musan: bool, skip_librispeech: bool) -> None:
    if not skip_vctk:
        logger.info("Downloading VCTK...")
        path = download_vctk(raw_dir)
        logger.info("VCTK ready at %s", path)

    if not skip_musan:
        logger.info("Downloading MUSAN (noise subset)...")
        path = download_musan_noise(raw_dir)
        logger.info("MUSAN noise ready at %s", path)

    if not skip_librispeech:
        logger.info("Downloading LibriSpeech dev-clean...")
        path = download_librispeech_dev_clean(raw_dir)
        logger.info("LibriSpeech dev-clean ready at %s", path)


if __name__ == "__main__":
    main()
