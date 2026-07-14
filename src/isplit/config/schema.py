"""Dataclass schema for I-SPLIT experiment configs. `pilot.yaml` and `full.yaml`
share this exact schema and differ only in scale-related field values, so every
downstream script is written once against these dataclasses regardless of scale.
"""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    vctk_dir: str = "data/raw/vctk"
    musan_dir: str = "data/raw/musan"
    librispeech_dir: str = "data/raw/librispeech_dev_clean"

    # pilot-scale caps; set to null/very large in full.yaml to use everything
    max_speakers: int | None = 20
    max_utterances_per_speaker: int | None = 15
    max_musan_noise_files: int | None = 40
    max_librispeech_speakers: int | None = 15
    max_librispeech_utterances_per_speaker: int | None = 10

    sample_rate: int = 16000
    held_out_speaker_fraction: float = 0.2

    n_speaker_pairs: int = 400
    n_content_pairs: int = 400
    n_environment_pairs: int = 400
    n_channel_pairs: int = 400
    snr_levels_db: list[float] = field(default_factory=lambda: [0.0, 10.0, 20.0])

    seed: int = 0


@dataclass
class EncoderConfig:
    name: str
    hf_id: str
    num_layers: int


@dataclass
class EncodersConfig:
    encoders: list[EncoderConfig] = field(
        default_factory=lambda: [
            EncoderConfig("wav2vec2-base", "facebook/wav2vec2-base", 12),
            EncoderConfig("hubert-base", "facebook/hubert-base-ls960", 12),
            EncoderConfig("wavlm-base", "microsoft/wavlm-base", 12),
            EncoderConfig("data2vec-audio-base", "facebook/data2vec-audio-base", 12),
        ]
    )
    pooling: str = "mean"  # utterance-level pooling for non-content probes
    batch_size: int = 4
    device: str = "cpu"
    # restrict a run to a subset of `encoders` by name (e.g. while iterating on
    # 2 of 4 encoders); None runs all of them
    active_encoders: list[str] | None = None

    def active(self) -> list[EncoderConfig]:
        if self.active_encoders is None:
            return self.encoders
        return [e for e in self.encoders if e.name in self.active_encoders]


@dataclass
class SubspaceConfig:
    rank: int | None = None  # None -> select via energy_threshold
    energy_threshold: float = 0.95
    tau_values: list[float] = field(default_factory=lambda: [1e-4, 1e-2, 1e-1, 1.0])
    factors: list[str] = field(default_factory=lambda: ["content", "speaker", "environment", "channel"])


@dataclass
class ExperimentConfig:
    scale: str = "pilot"
    data: DataConfig = field(default_factory=DataConfig)
    encoders: EncodersConfig = field(default_factory=EncodersConfig)
    subspace: SubspaceConfig = field(default_factory=SubspaceConfig)
    results_dir: str = "results"
    seed: int = 0
