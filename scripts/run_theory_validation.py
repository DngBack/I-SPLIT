"""Run the dependency-free synthetic validation of Propositions 1 and 2 and
save the resulting reports as CSV under results/tables/. No audio, no
downloads, no encoders needed -- this is the fastest correctness check of the
core I-SPLIT math and should be run first.

Usage: uv run python scripts/run_theory_validation.py
"""

from pathlib import Path

import click

from isplit.theory.validate_propositions import (
    prop1_eigenspace_identifiability,
    prop1_noise_sensitivity,
    prop2_oblique_vs_orthogonal,
)
from isplit.utils.logging import get_logger

logger = get_logger(__name__)


@click.command()
@click.option("--results-dir", default="results/tables", help="Where to write the report CSVs.")
def main(results_dir: str) -> None:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Proposition 1: eigenspace identifiability vs. sample size N")
    df1 = prop1_eigenspace_identifiability()
    df1.to_csv(out_dir / "prop1_identifiability_vs_n.csv", index=False)
    logger.info("\n%s", df1.to_string(index=False))

    logger.info("Proposition 1: recovery error vs. noise std")
    df1b = prop1_noise_sensitivity()
    df1b.to_csv(out_dir / "prop1_identifiability_vs_noise.csv", index=False)
    logger.info("\n%s", df1b.to_string(index=False))

    logger.info("Proposition 2: oblique vs. orthogonal separability vs. principal angle")
    df2 = prop2_oblique_vs_orthogonal()
    df2.to_csv(out_dir / "prop2_oblique_vs_orthogonal.csv", index=False)
    logger.info("\n%s", df2.to_string(index=False))

    logger.info("Wrote theory-validation reports to %s", out_dir)


if __name__ == "__main__":
    main()
