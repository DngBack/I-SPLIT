# I-SPLIT

**I-SPLIT** (Intervention-Guided Subspace Learning and Interchange Testing) audits
whether frozen self-supervised speech encoders (wav2vec2, HuBERT, WavLM,
data2vec-audio) are *causally separable* -- not just whether linguistic content,
speaker, environment, and channel information are linearly decodable or
geometrically orthogonal, but whether intervening on one factor's estimated
subspace actually changes that factor's downstream behavior while leaving
others intact.

This repo implements the full pipeline: paired-intervention subspace
estimation, regularized oblique projection, interchange-intervention causal
metrics (Preserve / Transfer / Causal Selectivity Score / Irreducible
Entanglement Index), baselines (PCA, orthogonal removal, probe-nullspace,
low-rank SVD, random subspace), and the experiment scripts that produce the
paper's tables and figures.

## Scope: pilot vs. full

This machine has **no GPU**. Every script is config-driven
(`configs/pilot.yaml` vs `configs/full.yaml`, identical schema, different
scale) so the same code runs a small CPU-feasible pilot now and scales to
paper-scale numbers later on a GPU machine with a config flag, not a rewrite.

**Results produced under `pilot.yaml` are preliminary pipeline validation, not
paper-final numbers.** Treat them as "does the method work end-to-end and
point in the right direction," not as reportable results.

## Setup

```bash
uv sync --dev
```

Pins Python 3.12.13 and installs a CPU-only PyTorch build via the
`pytorch-cpu` index configured in `pyproject.toml`.

## Known environment quirk (Windows)

On this development machine, the *first* import of any native-extension
package (`numpy`, `scipy`, `soundfile`, `sklearn`, `torch`) inside a freshly
spawned `uv run python ...` process can take anywhere from ~20 seconds to
several minutes -- almost certainly Windows Defender (or similar real-time AV)
scanning newly-touched DLLs, worse under concurrent I/O load (e.g. while a
large dataset download is also running). This is **not a code bug**. It:

- Is paid once per process, not per import statement after the first native one.
- Is why every script here is written as one long-running process (e.g.
  `extract_features.py` loops over all requested utterances in a single
  process) rather than spawning a new Python process per item.
- Means `pytest` runs and one-off scripts should be given generous timeouts
  (multiple minutes) rather than assumed hung. If a run is genuinely stuck
  (near-zero CPU time growth over several minutes via Task Manager /
  `Get-Process -Id <pid> | select CPU`), that's the actual signal something
  is wrong -- elapsed wall-clock time alone is not.

## Running the pipeline

See **[RUNBOOK.md](RUNBOOK.md)** for detailed, step-by-step server
instructions (GPU setup, resuming interrupted downloads, output locations,
troubleshooting). Quick version:

```bash
# 1. Validate the core math -- fast, no downloads, no audio
uv run python scripts/run_theory_validation.py

# 2. Download VCTK (~12GB), MUSAN noise subset, LibriSpeech dev-clean (~340MB)
uv run python scripts/download_data.py

# 3. Build manifests + intervention pairs, run all 4 frozen encoders,
#    cache per-layer hidden states (long-running, single process)
uv run python scripts/extract_features.py --scale pilot

# 4. Fit I-SPLIT intervention-covariance subspaces + PCA/random baselines
uv run python scripts/fit_subspaces.py --scale pilot

# 5. Train probes, run interchange interventions, build the layer-wise audit table
uv run python scripts/run_interchange_eval.py --scale pilot

# 6. Assemble paper tables/figures (Claim 1 correlation, ranking disagreement, plots)
uv run python scripts/make_paper_tables.py --scale pilot
```

Swap `--scale pilot` for `--scale full` (and `encoders.device: cuda` in
`configs/full.yaml`) once running on a GPU machine with the full downloads.

## Tests

```bash
uv run pytest
```

Priority order for the math-critical modules (highest bug-risk first, per the
project plan): `subspace/projection.py` (oblique ridge projection),
`subspace/intervention_covariance.py` (Proposition 1), `causal/interchange.py`
+ `causal/metrics.py` (swap formula, CSS), `data/pairs.py` (speaker-disjoint
leakage prevention), `encoders/cache.py` (round-trip/idempotency). All are
tested against analytically-known synthetic ground truth, not just "does it run."

## Data sources

- **VCTK Corpus v0.92**: https://datashare.ed.ac.uk/handle/10283/3443 -- ~110
  speakers reading identical fixed prompts, giving true same-text/
  different-speaker pairs (speaker factor) and same-speaker/different-text
  pairs (content factor).
- **MUSAN** (noise subset): primary source is the verified raw-audio HF mirror
  `FluidInference/musan`; falls back to https://www.openslr.org/resources/17/
  if the mirror is unavailable. Used for environment-factor noise overlay.
- **LibriSpeech dev-clean**: https://www.openslr.org/resources/12/ -- natural,
  non-synthetic held-out validation set (never used for subspace estimation).
- Channel-factor conditions (telephone-band filtering, mu-law companding) are
  synthesized via signal processing (`scipy.signal` + a hand-rolled mu-law
  codec) -- no additional dataset needed.
