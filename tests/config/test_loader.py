from isplit.config.loader import load_config


def test_load_pilot_config_from_real_configs_dir():
    cfg = load_config(scale="pilot", config_dir="configs")
    assert cfg.scale == "pilot"
    assert cfg.data.max_speakers == 40
    assert cfg.data.max_utterances_per_speaker == 20
    # device is a property of the machine the pilot runs on, not of the scale
    # (the RUNBOOK has you set it to cuda for a GPU-accelerated pilot), so pin
    # the merge behaviour rather than one host's answer.
    assert cfg.encoders.device in ("cpu", "cuda")
    assert len(cfg.encoders.encoders) == 4
    # active_encoders narrows which of the 4 registered encoders a run actually
    # touches; pilot.yaml currently runs all of them (None), but the field
    # exists to subset -- see EncodersConfig.active().
    assert cfg.encoders.active_encoders is None
    assert [e.name for e in cfg.encoders.active()] == [e.name for e in cfg.encoders.encoders]


def test_load_full_config_has_no_speaker_cap():
    cfg = load_config(scale="full", config_dir="configs")
    assert cfg.scale == "full"
    assert cfg.data.max_speakers is None
    assert cfg.data.n_speaker_pairs == 20000


def test_pilot_and_full_share_schema_only_differ_in_scale_fields():
    pilot = load_config(scale="pilot", config_dir="configs")
    full = load_config(scale="full", config_dir="configs")
    assert pilot.subspace.energy_threshold == full.subspace.energy_threshold
    assert pilot.results_dir == full.results_dir
    assert pilot.data.n_speaker_pairs != full.data.n_speaker_pairs


def test_load_config_missing_scale_file_falls_back_to_base(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 42\n", encoding="utf-8")
    cfg = load_config(scale="nonexistent", config_dir=tmp_path)
    assert cfg.seed == 42
    assert cfg.scale == "nonexistent"
