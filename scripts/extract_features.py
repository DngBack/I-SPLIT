"""Build manifests + intervention pairs from the downloaded raw data, then run
every frozen encoder over exactly the (utterance, condition) combinations the
pairs need, caching per-layer hidden states to results/features/.

This is a single long-running process on purpose: importing torch/transformers/
soundfile/scipy has a large one-time cost on this machine, so we pay it once
here rather than once per utterance or per script invocation.

Usage: uv run python scripts/extract_features.py --scale pilot
"""

from pathlib import Path

import click
import pandas as pd

from isplit.config.loader import load_config
from isplit.data.augment import apply_condition
from isplit.data.io import load_wav
from isplit.data.manifest import (
    apply_file_cap,
    apply_pilot_caps,
    build_librispeech_manifest,
    build_musan_noise_manifest,
    build_vctk_manifest,
)
from isplit.data.pairs import build_all_pairs, split_speakers
from isplit.data.requests import collect_feature_requests
from isplit.encoders.cache import cache_path, has_cache, save_features
from isplit.encoders.extract import extract_hidden_states
from isplit.encoders.registry import load_encoder
from isplit.utils.logging import get_logger
from isplit.utils.seeding import set_all_seeds

logger = get_logger(__name__)


@click.command()
@click.option("--scale", default="pilot", type=click.Choice(["pilot", "full"]))
@click.option("--config-dir", default="configs")
def main(scale: str, config_dir: str) -> None:
    cfg = load_config(scale=scale, config_dir=config_dir)
    set_all_seeds(cfg.seed)

    manifests_dir = Path(cfg.results_dir) / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(cfg.results_dir) / "features"

    logger.info("Building VCTK manifest from %s", cfg.data.vctk_dir)
    vctk = build_vctk_manifest(cfg.data.vctk_dir)
    vctk = apply_pilot_caps(
        vctk, cfg.data.max_speakers, cfg.data.max_utterances_per_speaker, seed=cfg.data.seed
    )
    logger.info("VCTK manifest: %d utterances, %d speakers", len(vctk), vctk["speaker_id"].nunique())
    vctk.to_parquet(manifests_dir / "vctk.parquet")

    logger.info("Building MUSAN noise manifest from %s", cfg.data.musan_dir)
    noise = build_musan_noise_manifest(cfg.data.musan_dir)
    noise = apply_file_cap(noise, cfg.data.max_musan_noise_files, seed=cfg.data.seed)
    logger.info("MUSAN noise manifest: %d clips", len(noise))
    noise.to_parquet(manifests_dir / "musan_noise.parquet")

    logger.info("Building intervention pairs...")
    n_pairs = {
        "speaker": cfg.data.n_speaker_pairs,
        "content": cfg.data.n_content_pairs,
        "environment": cfg.data.n_environment_pairs,
        "channel": cfg.data.n_channel_pairs,
    }
    pair_sets = build_all_pairs(
        vctk, noise, cfg.data.snr_levels_db, n_pairs, cfg.data.held_out_speaker_fraction, cfg.data.seed
    )
    for factor, df in pair_sets.items():
        logger.info("%s pairs: %d (train=%d, held_out=%d)", factor, len(df),
                    (df["split"] == "train").sum() if not df.empty else 0,
                    (df["split"] == "held_out").sum() if not df.empty else 0)
        df.to_parquet(manifests_dir / f"pairs_{factor}.parquet")

    requests = collect_feature_requests(pair_sets)

    # The pair tables only cover utterances that happened to be sampled into a
    # pair, but the probes and the decodability audit (eval/layerwise_audit.py)
    # read clean features for *every* manifest utterance. Request those too, on
    # the same speaker-disjoint split the pairs were built against.
    train_speakers, _ = split_speakers(
        vctk["speaker_id"], cfg.data.held_out_speaker_fraction, cfg.data.seed
    )
    clean_requests = pd.DataFrame(
        {
            "utt_id": vctk["utt_id"],
            "condition": None,
            "split": [
                "train" if sid in train_speakers else "held_out" for sid in vctk["speaker_id"]
            ],
        }
    )
    requests = (
        pd.concat([requests, clean_requests], ignore_index=True)
        .drop_duplicates(subset=["utt_id", "condition"])
        .reset_index(drop=True)
    )
    logger.info("Total VCTK (utterance, condition) feature requests: %d", len(requests))

    logger.info("Building LibriSpeech dev-clean manifest from %s", cfg.data.librispeech_dir)
    librispeech = build_librispeech_manifest(cfg.data.librispeech_dir)
    librispeech = apply_pilot_caps(
        librispeech,
        cfg.data.max_librispeech_speakers,
        cfg.data.max_librispeech_utterances_per_speaker,
        seed=cfg.data.seed,
    )
    logger.info(
        "LibriSpeech manifest: %d utterances, %d speakers", len(librispeech), librispeech["speaker_id"].nunique()
    )
    librispeech.to_parquet(manifests_dir / "librispeech.parquet")
    # LibriSpeech is natural held-out validation data -- never used for subspace
    # estimation, always cached under the 'held_out' split.
    librispeech_requests = pd.DataFrame(
        {"utt_id": librispeech["utt_id"], "condition": None, "split": "held_out"}
    )

    requests = pd.concat([requests, librispeech_requests], ignore_index=True)
    logger.info("Total feature requests (VCTK + LibriSpeech): %d", len(requests))
    requests.to_parquet(manifests_dir / "feature_requests.parquet")

    utt_lookup = pd.concat([vctk, librispeech], ignore_index=True).set_index("utt_id")
    utt_lookup = utt_lookup[~utt_lookup.index.duplicated(keep="first")]
    noise_wav_cache: dict[str, tuple] = {}

    def _get_noise_wav(noise_id: str, sr: int):
        key = (noise_id, sr)
        if key not in noise_wav_cache:
            wav_path = noise.set_index("noise_id").loc[noise_id, "wav_path"]
            wav, _ = load_wav(wav_path, target_sr=sr)
            noise_wav_cache[key] = wav
        return noise_wav_cache[key]

    for encoder_cfg in cfg.encoders.active():
        logger.info("Loading encoder %s (%s)...", encoder_cfg.name, encoder_cfg.hf_id)
        spec = load_encoder(encoder_cfg.name, device=cfg.encoders.device)

        n_done, n_skipped = 0, 0
        for _, req in requests.iterrows():
            utt_id, condition, split = req["utt_id"], req["condition"], req["split"]
            path = cache_path(features_dir, encoder_cfg.name, split, utt_id, condition)
            if has_cache(path):
                n_skipped += 1
                continue

            wav_path = utt_lookup.loc[utt_id, "wav_path"]
            wav, sr = load_wav(wav_path, target_sr=16000)

            if condition is not None and condition != "clean":
                noise_lookup = None
                if condition.startswith("noise="):
                    noise_id = condition[len("noise=") :].split("_snr=")[0]
                    noise_lookup = {noise_id: _get_noise_wav(noise_id, sr)}
                wav = apply_condition(wav, sr, condition, noise_lookup=noise_lookup, seed=cfg.data.seed)

            features = extract_hidden_states(spec, wav, sr, device=cfg.encoders.device)
            save_features(path, features)
            n_done += 1
            if n_done % 50 == 0:
                logger.info("%s: extracted %d, skipped(cached) %d / %d", encoder_cfg.name, n_done, n_skipped, len(requests))

        logger.info("%s: done. extracted=%d skipped=%d total=%d", encoder_cfg.name, n_done, n_skipped, len(requests))


if __name__ == "__main__":
    main()
