import numpy as np

from isplit.data.augment import (
    apply_condition,
    bandpass_telephone,
    encode_channel_condition,
    encode_environment_condition,
    measured_snr_db,
    mulaw_roundtrip,
    overlay_noise,
)
from isplit.data.io import resample


def _tone(freq, sr, duration_s, amp=0.3):
    t = np.arange(int(sr * duration_s)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_overlay_noise_achieves_target_snr_within_tolerance():
    rng = np.random.default_rng(0)
    sr = 16000
    wav = _tone(220, sr, 2.0)
    noise = rng.standard_normal(sr * 4).astype(np.float32)

    for target_snr in (0.0, 10.0, 20.0):
        noisy = overlay_noise(wav, noise, snr_db=target_snr, seed=1)
        achieved = measured_snr_db(wav, noisy)
        assert abs(achieved - target_snr) < 0.5


def test_overlay_noise_handles_short_noise_by_looping():
    wav = _tone(220, 16000, 2.0)
    short_noise = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    noisy = overlay_noise(wav, short_noise, snr_db=10.0, seed=0)
    assert noisy.shape == wav.shape
    assert np.all(np.isfinite(noisy))


def test_overlay_noise_zero_noise_returns_clean_copy():
    wav = _tone(220, 16000, 1.0)
    silent_noise = np.zeros(16000, dtype=np.float32)
    noisy = overlay_noise(wav, silent_noise, snr_db=10.0, seed=0)
    assert np.allclose(noisy, wav)


def test_bandpass_telephone_attenuates_out_of_band_energy():
    sr = 16000
    low_tone = _tone(100, sr, 1.0)  # below the 300Hz telephone band
    in_band_tone = _tone(1000, sr, 1.0)  # within the 300-3400Hz band

    filtered_low = bandpass_telephone(low_tone, sr)
    filtered_in_band = bandpass_telephone(in_band_tone, sr)

    low_attenuation = np.sqrt(np.mean(filtered_low**2)) / np.sqrt(np.mean(low_tone**2))
    in_band_attenuation = np.sqrt(np.mean(filtered_in_band**2)) / np.sqrt(np.mean(in_band_tone**2))

    assert low_attenuation < 0.3
    assert in_band_attenuation > 0.7


def test_bandpass_telephone_preserves_shape():
    wav = _tone(500, 16000, 1.5)
    filtered = bandpass_telephone(wav, 16000)
    assert filtered.shape == wav.shape


def test_mulaw_roundtrip_approximately_identity_for_moderate_amplitude():
    wav = _tone(300, 16000, 1.0, amp=0.5)
    roundtripped = mulaw_roundtrip(wav)
    assert roundtripped.shape == wav.shape
    correlation = np.corrcoef(wav, roundtripped)[0, 1]
    assert correlation > 0.99


def test_mulaw_roundtrip_bounded_output():
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(16000).astype(np.float32)
    out = mulaw_roundtrip(wav)
    assert np.all(np.abs(out) <= 1.0 + 1e-6)


def test_resample_preserves_duration_and_energy_roughly():
    sr_from, sr_to = 48000, 16000
    wav = _tone(220, sr_from, 1.0)
    resampled = resample(wav, sr_from, sr_to)

    expected_len = round(len(wav) * sr_to / sr_from)
    assert abs(len(resampled) - expected_len) <= 2

    rms_before = np.sqrt(np.mean(wav**2))
    rms_after = np.sqrt(np.mean(resampled**2))
    assert abs(rms_before - rms_after) / rms_before < 0.1


def test_resample_identity_when_rates_match():
    wav = _tone(220, 16000, 1.0)
    out = resample(wav, 16000, 16000)
    assert np.array_equal(out, wav)


def test_apply_condition_clean_is_noop():
    wav = _tone(220, 16000, 1.0)
    assert np.array_equal(apply_condition(wav, 16000, None), wav)
    assert np.array_equal(apply_condition(wav, 16000, "clean"), wav)


def test_apply_condition_environment_matches_direct_overlay_noise():
    wav = _tone(220, 16000, 1.0)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(16000).astype(np.float32)
    condition = encode_environment_condition("n1", 10.0)

    via_condition = apply_condition(wav, 16000, condition, noise_lookup={"n1": noise}, seed=0)
    via_direct = overlay_noise(wav, noise, snr_db=10.0, seed=0)

    assert np.allclose(via_condition, via_direct)


def test_apply_condition_environment_missing_noise_raises():
    wav = _tone(220, 16000, 1.0)
    condition = encode_environment_condition("missing", 10.0)
    try:
        apply_condition(wav, 16000, condition, noise_lookup={})
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_apply_condition_channel_telephone_matches_direct():
    wav = _tone(220, 16000, 1.0)
    condition = encode_channel_condition("telephone")
    via_condition = apply_condition(wav, 16000, condition)
    via_direct = bandpass_telephone(wav, 16000)
    assert np.allclose(via_condition, via_direct)


def test_apply_condition_channel_mulaw_matches_direct():
    wav = _tone(220, 16000, 1.0)
    condition = encode_channel_condition("mulaw")
    via_condition = apply_condition(wav, 16000, condition)
    via_direct = mulaw_roundtrip(wav)
    assert np.allclose(via_condition, via_direct)


def test_apply_condition_unknown_encoding_raises():
    wav = _tone(220, 16000, 1.0)
    try:
        apply_condition(wav, 16000, "bogus=xyz")
        raised = False
    except ValueError:
        raised = True
    assert raised
