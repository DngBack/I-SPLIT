# RUNBOOK — running I-SPLIT end-to-end on a server

Step-by-step instructions for taking this repo to a server (ideally with a
GPU) and running the full pipeline. Every command below is run from the repo
root.

## 0. Prerequisites

- Python 3.12 (uv will install/pin this automatically, no manual install needed)
- `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh` (Linux/Mac) or see https://docs.astral.sh/uv/getting-started/installation/
- Disk space: ~15GB for pilot scale (VCTK + MUSAN noise subset + LibriSpeech
  dev-clean + cached features), 30GB+ recommended headroom. Full scale needs more.
- (Optional but recommended) an NVIDIA GPU + drivers, for `--scale full` or
  faster pilot runs. CPU-only works fine for pilot scale, just slower.

If you're copying this exact working directory to the server (not a fresh
`git clone`), `data/raw/downloads/VCTK-Corpus-0.92.zip` already has a partial
download (~3.9GB of 11.7GB) that will **resume automatically** rather than
restart, since `scripts/download_data.py` uses HTTP Range-based resume.

## 1. Install dependencies

```bash
uv sync --dev
```

Installs Python 3.12.13 (pinned via `.python-version`) and a **CPU-only**
PyTorch build (see `pyproject.toml`'s `[tool.uv.sources]` pinning the
`pytorch-cpu` wheel index).

### If the server has a GPU

Edit `pyproject.toml` before running `uv sync`:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"          # rename or add a new index, e.g. "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"   # pick the CUDA version matching the server's driver
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]   # point at the renamed/new index above
```

Then `uv sync --dev`, and set `encoders.device: cuda` in `configs/full.yaml`
(or `configs/pilot.yaml`, if you want a GPU-accelerated pilot run).

## 2. Sanity-check the core math (fast, no data/network needed)

```bash
uv run python scripts/run_theory_validation.py
uv run pytest
```

`run_theory_validation.py` empirically validates the two theoretical claims
(Proposition 1: intervention-covariance eigenspace identifiability;
Proposition 2: oblique-vs-orthogonal separability) on synthetic data and
writes CSVs to `results/tables/`. `pytest` runs the full test suite (~190
tests). Both should be fast on a normal server; on the Windows dev machine
this was built on, first-time imports of `numpy`/`scipy`/`soundfile`/`torch`
were anomalously slow (tens of seconds to minutes each, likely antivirus
real-time scanning) — if the same happens on your server's first run, that's
a one-time cost per fresh environment, not a code issue.

## 3. Download the datasets

```bash
uv run python scripts/download_data.py
```

Downloads, in order:
1. **VCTK Corpus v0.92** (~11.7GB zip) from the official Edinburgh DataShare host.
2. **MUSAN noise subset** — tries a verified HF raw-audio mirror first
   (`FluidInference/musan`, ~585MB total, only the `noise/` folder is
   fetched), falls back to the official openslr.org full tarball (~11GB) if
   the mirror is unreachable.
3. **LibriSpeech dev-clean** (~337MB) from openslr.org.

Useful flags:
- `--raw-dir /path/to/data` — download elsewhere (default `data/raw`)
- `--skip-vctk` / `--skip-musan` / `--skip-librispeech` — skip one dataset
- Safe to re-run / interrupt and resume: downloads resume via HTTP Range,
  and extraction is skipped for anything already fully extracted (marked by
  a `.done` file per dataset directory — a crash mid-extraction is detected
  and retried automatically, not silently treated as complete).

## 4. Extract frozen-encoder features (the slow step)

```bash
uv run python scripts/extract_features.py --scale pilot
```

This single long-running process:
- builds manifests for VCTK / MUSAN / LibriSpeech from the raw downloads,
- builds all 4 factor intervention pair tables (content, speaker,
  environment, channel) with a speaker-disjoint train/held-out split,
- runs all 4 frozen encoders (wav2vec2-base, HuBERT-base, WavLM-base,
  data2vec-audio-base) over every needed (utterance, condition) combination,
  caching per-layer hidden states to `results/features/`.

It's idempotent — already-cached (encoder, utterance, condition) combos are
skipped on re-run, so if it's interrupted, just re-run the same command.

## 5. Fit I-SPLIT subspaces + baselines

```bash
uv run python scripts/fit_subspaces.py --scale pilot
```

For every (encoder, layer, factor): estimates the intervention-covariance
subspace (I-SPLIT), plus PCA and random-subspace baselines. Writes `.npz`
bases to `results/subspaces/` and a summary to
`results/tables/subspace_fit_summary.csv`.

## 6. Run the causal interchange evaluation

```bash
uv run python scripts/run_interchange_eval.py --scale pilot
```

Trains linear probes (speaker-id, content/prompt-id), runs interchange
interventions using the fitted subspaces, and computes decodability / IEI /
Causal Selectivity Score per (encoder, layer, factor). Writes
`results/tables/layerwise_audit.csv`.

## 7. Assemble the paper tables/figures

```bash
uv run python scripts/make_paper_tables.py --scale pilot
```

Produces the Claim-1 evidence: `results/tables/claim1_correlation.csv`
(probe-vs-CSS Spearman correlation), `claim1_ranking_disagreement.csv`
(layers where decodability and CSS rank differently), and
`results/figures/layerwise_audit.png`.

## Switching pilot → full scale

Replace `--scale pilot` with `--scale full` in steps 4-7. `configs/full.yaml`
already removes the speaker/utterance caps and uses larger pair counts; bump
`encoders.batch_size` and set `encoders.device: cuda` there once running on a
GPU. No code changes needed — the config is the only thing that changes.

## Output map

```
results/
  tables/       all CSV metric tables (theory validation, subspace fit
                summary, layer-wise audit, Claim 1 correlation/ranking)
  figures/      layerwise_audit.png (extend make_paper_tables.py for more)
  features/     cached per-utterance encoder hidden states (large, gitignored)
  subspaces/    fitted subspace bases per (encoder, layer, factor, method)
  manifests/    VCTK/MUSAN/LibriSpeech manifests + pair tables (parquet)
```

## Troubleshooting

- **A script fails partway through**: safe to just re-run it. Dataset
  extraction, feature caching, and subspace fitting are all designed to skip
  already-completed work.
- **Disk pressure**: `configs/pilot.yaml`'s `max_speakers` /
  `max_utterances_per_speaker` / `max_musan_noise_files` /
  `max_librispeech_*` caps control how much of the *downloaded* data actually
  gets processed — you can download once and iterate on caps without
  re-downloading.
- **Ablations not yet wired into a script**: `src/isplit/eval/ablations.py`
  has the tested library functions (true-vs-mismatched pairs, isolated-vs-
  joint nuisance discovery, linear-vs-nonlinear leakage, gold-vs-pseudo pair
  labels) but no dedicated CLI script yet — call them directly from a short
  Python script or notebook once `results/features/` and `results/subspaces/`
  are populated, following the same loading pattern as
  `scripts/run_interchange_eval.py`.
