"""Command-line interface: ``third-allele-errors BAM REFERENCE [options]``."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import defopt
import polars as pl

from .estimator import ThirdAlleleErrorEstimator, truncation_loss

log = logging.getLogger("third_allele_errors")


def main(
    bam: Path,
    reference: Path,
    *,
    output: Optional[Path] = None,
    sites_output: Optional[Path] = None,
    regions: Optional[str] = None,
    min_mapq: int = 20,
    min_baseq: int = 20,
    max_depth: int = 10_000,
    min_qual: float = 30.0,
    min_depth: int = 20,
    max_site_depth: Optional[int] = None,
    min_het_vaf: float = 0.3,
    max_het_vaf: float = 0.7,
    max_errors_per_site: Optional[int] = None,
    artefact_model: bool = True,
    artefact_min_reads: int = 3,
    artefact_tail_quantile: float = 0.999,
    artefact_max_vaf: float = 0.5,
    prior_error_rate: Optional[float] = None,
    max_em_iterations: int = 1000,
    lod_depth: Optional[int] = None,
    alpha: float = 0.05,
    power: float = 0.95,
    threads: int = 0,
) -> None:
    """Estimate sequencing error rates from third alleles at germline het SNPs.

    :param bam: Indexed BAM/CRAM for a single sample.
    :param reference: Indexed reference FASTA.
    :param output: Write the error table as TSV here (always printed to stdout).
    :param sites_output: Write the per-site table (with QC flags and artefact
        weights) as TSV here.
    :param regions: Region string (chr1:1-1e6) or BED file to restrict the pileup.
    :param min_mapq: Minimum read mapping quality.
    :param min_baseq: Minimum base quality.
    :param max_depth: Max per-file depth for bcftools mpileup.
    :param min_qual: Minimum QUAL for a het call.
    :param min_depth: Minimum depth at a het site.
    :param max_site_depth: Maximum depth at a het site.
    :param min_het_vaf: Minimum germline allele fraction for a het site.
    :param max_het_vaf: Maximum germline allele fraction for a het site.
    :param max_errors_per_site: Optional hard cap: drop sites with more
        third-allele reads than this (legacy; biases rates low).
    :param artefact_model: Down-weight likely artefacts (mapping errors,
        multi-allelic sites, contamination) with an error/artefact mixture.
    :param artefact_min_reads: Minimum read-count floor of the artefact component.
    :param artefact_tail_quantile: The artefact floor is at least this quantile
        of the expected error count at the site.
    :param artefact_max_vaf: Upper bound of the artefact VAF prior.
    :param prior_error_rate: Starting per-substitution error rate (default: the
        one-error-per-site estimate).
    :param max_em_iterations: Maximum EM iterations per artefact-floor round.
    :param lod_depth: Depth for the LoD estimate (default: median depth of used sites).
    :param alpha: False-positive rate for the LoD read threshold (limit of blank).
    :param power: Detection probability that defines the LoD (0.95 = LoD95).
    :param threads: Extra bcftools compression threads.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    est = ThirdAlleleErrorEstimator(
        bam=bam,
        reference=reference,
        regions=regions,
        min_mapq=min_mapq,
        min_baseq=min_baseq,
        max_depth=max_depth,
        min_qual=min_qual,
        min_depth=min_depth,
        max_site_depth=max_site_depth,
        min_het_vaf=min_het_vaf,
        max_het_vaf=max_het_vaf,
        max_errors_per_site=max_errors_per_site,
        artefact_model=artefact_model,
        artefact_min_reads=artefact_min_reads,
        artefact_tail_quantile=artefact_tail_quantile,
        artefact_max_vaf=artefact_max_vaf,
        prior_error_rate=prior_error_rate,
        max_em_iterations=max_em_iterations,
        threads=threads,
    )
    table = est.error_table(lod_depth=lod_depth, alpha=alpha, power=power)

    summary = est.summary()
    for key, val in summary.items():
        log.info("%-20s %s", key, f"{val:.4g}" if isinstance(val, float) else val)
    if summary["used"] and max_errors_per_site is not None:
        mean_kept = est.used_sites["n_err"].mean()
        lost = truncation_loss(mean_kept, max_errors_per_site)
        log.info("%-20s %.1f%%", "est_errors_lost", 100 * lost)
        if math.isnan(lost) or lost > 0.05:
            log.warning(
                "The --max-errors-per-site=%d cap discards about %.0f%% of error "
                "reads, biasing rates low (most for the highest-rate types). "
                "Consider dropping it and relying on the artefact model.",
                max_errors_per_site,
                100 * lost,
            )
    if summary["used"] and artefact_model:
        bias = est.rate_bias
        log.info("%-20s %+.2f%%", "est_rate_bias", 100 * bias)
        if bias < -0.05:
            log.warning(
                "Artefact weighting is expected to absorb enough genuine errors "
                "to bias rates low by about %.0f%%. Consider raising "
                "--artefact-tail-quantile.",
                -100 * bias,
            )

    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_float="full", tbl_width_chars=200):
        print(
            table.with_columns(
                pl.col("errors", "opportunities").round(1),
                pl.col("error_rate", "ci_low", "ci_high").map_elements(
                    lambda x: f"{x:.3e}", return_dtype=pl.Utf8
                ),
                pl.col("lod_vaf").map_elements(
                    lambda x: f"{x:.4%}", return_dtype=pl.Utf8
                ),
            )
        )

    if output:
        table.write_csv(output, separator="\t")
    if sites_output:
        est.sites_report().write_csv(sites_output, separator="\t")


def run() -> None:
    """Console-script entry point."""
    defopt.run(main)
