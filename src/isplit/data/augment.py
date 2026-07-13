"""DSP-only augmentation for synthesizing environment/channel intervention
pairs from a single clean utterance: noise overlay at a target SNR (environment
factor) and telephone-band filtering / mu-law companding (channel factor). No
extra dataset needed for the channel factor; environment overlay draws noise
clips from the MUSAN manifest.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def _rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(wav.astype(np.float64) ** 2) + 1e-12))


def overlay_noise(wav: np.ndarray, noise: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Overlay `noise` onto `wav` at the target SNR (dB), looping/cropping the
    noise clip to match `wav`'s length. A random crop offset (seeded) is used
    when the noise clip is longer than `wav`.
    """
    rng = np.random.default_rng(seed)
    n = len(wav)
    if len(noise) < n:
        reps = int(np.ceil(n / max(len(noise), 1)))
        noise = np.tile(noise, reps)
    offset = 0 if len(noise) == n else int(rng.integers(0, len(noise) - n + 1))
    noise_seg = noise[offset : offset + n]

    speech_rms = _rms(wav)
    noise_rms = _rms(noise_seg)
    if noise_rms < 1e-8:
        return wav.copy()
    target_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    noise_scaled = noise_seg * (target_noise_rms / noise_rms)
    return (wav + noise_scaled).astype(np.float32)


def measured_snr_db(wav: np.ndarray, noisy: np.ndarray) -> float:
    """Measure the SNR actually achieved by `overlay_noise`, for testing/QA:
    treats (noisy - wav) as the injected noise and compares RMS to the clean signal.
    """
    injected_noise = noisy - wav
    speech_rms = _rms(wav)
    noise_rms = _rms(injected_noise)
    if noise_rms < 1e-8:
        return float("inf")
    return float(20 * np.log10(speech_rms / noise_rms))


def bandpass_telephone(wav: np.ndarray, sr: int, low_hz: float = 300.0, high_hz: float = 3400.0) -> np.ndarray:
    """Simulate a telephone-band channel: 4th-order Butterworth bandpass,
    zero-phase (filtfilt) to avoid introducing group delay.
    """
    nyquist = sr / 2.0
    sos = butter(4, [low_hz / nyquist, min(high_hz / nyquist, 0.999)], btype="bandpass", output="sos")
    return sosfiltfilt(sos, wav).astype(np.float32)


def mulaw_roundtrip(wav: np.ndarray, mu: int = 255) -> np.ndarray:
    """Mu-law encode then decode, simulating a lossy compressed-channel codec."""
    x = np.clip(wav, -1.0, 1.0).astype(np.float64)
    encoded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quantized = np.round((encoded + 1) / 2 * mu) / mu * 2 - 1
    decoded = np.sign(quantized) * (np.expm1(np.abs(quantized) * np.log1p(mu))) / mu
    return decoded.astype(np.float32)


def encode_environment_condition(noise_id: str | None, snr_db: float | None) -> str:
    if noise_id is None or noise_id == "clean":
        return "clean"
    return f"noise={noise_id}_snr={snr_db}"


def encode_channel_condition(channel: str) -> str:
    return f"channel={channel}"


def apply_condition(
    wav: np.ndarray,
    sr: int,
    condition: str | None,
    noise_lookup: dict[str, np.ndarray] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Apply an environment/channel condition string (as produced by
    `encode_environment_condition` / `encode_channel_condition`) to a clean
    waveform. `condition=None` or `"clean"` is a no-op.
    """
    if condition is None or condition == "clean":
        return wav
    if condition.startswith("noise="):
        body = condition[len("noise=") :]
        noise_id, _, snr_part = body.partition("_snr=")
        if noise_lookup is None or noise_id not in noise_lookup:
            raise ValueError(f"no noise waveform provided for noise_id={noise_id!r}")
        return overlay_noise(wav, noise_lookup[noise_id], snr_db=float(snr_part), seed=seed)
    if condition.startswith("channel="):
        channel = condition[len("channel=") :]
        if channel == "telephone":
            return bandpass_telephone(wav, sr)
        if channel == "mulaw":
            return mulaw_roundtrip(wav)
        if channel == "clean":
            return wav
        raise ValueError(f"unknown channel condition: {channel!r}")
    raise ValueError(f"unrecognized condition encoding: {condition!r}")
