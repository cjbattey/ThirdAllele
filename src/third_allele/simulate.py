"""Simulators with known error rates and artefacts, for tests and benchmarks.

* :func:`simulate_sites` builds the het-site table that ``bcftools query``
  would emit, so the estimator can be exercised without bcftools.
* :func:`simulate_reads` writes a reference FASTA and a coordinate-sorted SAM,
  for end-to-end runs through bcftools.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .estimator import MUTATION_TYPES, QUERY_COLUMNS

BASES = np.array(list("ACGT"))
_COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}

#: Relative error rate per folded class; scaled so its mean is the target rate.
DEFAULT_PROFILE = {
    "C>A": 0.8,
    "C>G": 0.3,
    "C>T": 2.0,
    "T>A": 0.4,
    "T>C": 1.0,
    "T>G": 0.5,
}


def _fold(src: str, dst: str) -> str:
    return f"{_COMP[src]}>{_COMP[dst]}" if src in "AG" else f"{src}>{dst}"


def true_rates(rate: float, profile: dict[str, float] = DEFAULT_PROFILE) -> dict[str, float]:
    """Per-class rates for a mean per-substitution ``rate``, plus ``overall``."""
    scale = rate / np.mean([profile[t] for t in MUTATION_TYPES])
    rates = {t: profile[t] * scale for t in MUTATION_TYPES}
    rates["overall"] = rate
    return rates


def _rate_matrix(rate: float, profile: dict[str, float]) -> np.ndarray:
    """4x4 matrix of P(read base = dst | true base = src)."""
    rates = true_rates(rate, profile)
    m = np.zeros((4, 4))
    for i, s in enumerate(BASES):
        for j, d in enumerate(BASES):
            if i != j:
                m[i, j] = rates[_fold(s, d)]
    return m


def _third_alleles(ref: np.ndarray, alt: np.ndarray) -> np.ndarray:
    """The two bases at each site that are neither ref nor alt, shape (n, 2)."""
    mask = np.ones((ref.size, 4), dtype=bool)
    mask[np.arange(ref.size), ref] = False
    mask[np.arange(ref.size), alt] = False
    return np.nonzero(mask)[1].reshape(-1, 2)


@dataclass
class SiteSimulation:
    raw_sites: pl.DataFrame  # same columns as ThirdAlleleErrorEstimator.raw_sites
    truth: dict[str, float]  # true rate per class, plus "overall"
    is_artefact: np.ndarray  # per site


def simulate_sites(
    n: int,
    depth: float,
    rate: float,
    artefact_fraction: float,
    artefact_vaf: tuple[float, float] = (0.005, 0.2),
    profile: dict[str, float] = DEFAULT_PROFILE,
    seed: int = 0,
) -> SiteSimulation:
    """Simulate ``n`` germline het sites as a ``bcftools query`` table.

    Depth is negative binomial around ``depth`` (min 5). Each germline read
    errs to each other base at its class rate. A fraction of sites also carry
    an artefact: one third allele at a VAF drawn uniformly from ``artefact_vaf``.
    """
    rng = np.random.default_rng(seed)
    m = _rate_matrix(rate, profile)
    idx = np.arange(n)
    ref = rng.integers(0, 4, n)
    alt = (ref + rng.integers(1, 4, n)) % 4
    d = np.maximum(rng.negative_binomial(20, 20 / (20 + depth), n), 5)
    n_ref = rng.binomial(d, 0.5)

    counts = np.zeros((n, 4), dtype=np.int64)
    for src, n_src in [(ref, n_ref), (alt, d - n_ref)]:
        counts[idx, src] += n_src
        for b in range(4):
            e = rng.binomial(n_src, m[src, b])
            counts[idx, src] -= e
            counts[idx, b] += e

    third = _third_alleles(ref, alt)
    is_art = rng.random(n) < artefact_fraction
    x = third[idx, rng.integers(0, 2, n)]
    f = rng.uniform(*artefact_vaf, n)
    counts[idx[is_art], x[is_art]] += rng.binomial(d[is_art], f[is_art])

    rows = []
    for i in range(n):
        alts = [alt[i], *(b for b in third[i] if counts[i, b] > 0)]
        rows.append((
            "chr1", i + 1, BASES[ref[i]], ",".join(BASES[alts]), "100", "0/1",
            ",".join(str(counts[i, b]) for b in [ref[i], *alts]),
        ))  # fmt: skip
    df = pl.DataFrame(rows, schema=QUERY_COLUMNS, orient="row").with_columns(
        pl.col("pos").cast(pl.Int64)
    )
    return SiteSimulation(df, true_rates(rate, profile), is_art)


@dataclass
class ReadSimulation:
    reference: Path
    reads: Path  # coordinate-sorted SAM
    truth: dict[str, float]
    het_positions: np.ndarray  # 1-based
    artefact_positions: np.ndarray  # 1-based


def simulate_reads(
    outdir: Path,
    length: int = 100_000,
    depth: int = 200,
    read_length: int = 150,
    n_het: int = 500,
    n_artefact: int = 10,
    rate: float = 2e-3,
    artefact_vaf: tuple[float, float] = (0.05, 0.2),
    profile: dict[str, float] = DEFAULT_PROFILE,
    seed: int = 0,
) -> ReadSimulation:
    """Write ``ref.fa`` and a sorted single-end ``reads.sam`` into ``outdir``.

    Reads come from one of two haplotypes that differ at ``n_het`` sites (at
    least 150 bp apart). Every base errs at its class rate, and ``n_artefact``
    of the het sites get a third allele at a VAF drawn from ``artefact_vaf``.
    All bases are Q30 and all reads MAPQ 60. A sorted SAM needs no index for a
    whole-genome bcftools mpileup.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    m = _rate_matrix(rate, profile)

    ref = rng.integers(0, 4, length).astype(np.uint8)
    fasta = outdir / "ref.fa"
    with open(fasta, "w") as fh:
        fh.write(">chr1\n")
        s = "".join(BASES[ref])
        for i in range(0, length, 60):
            fh.write(s[i : i + 60] + "\n")

    margin = read_length
    het = np.sort(
        rng.choice(np.arange(margin, length - margin, 150), n_het, replace=False)
    )
    hap1 = ref.copy()
    hap1[het] = (ref[het] + rng.integers(1, 4, n_het)) % 4
    art = np.sort(rng.choice(het, n_artefact, replace=False))
    art_x = _third_alleles(ref[art], hap1[art])[np.arange(n_artefact), rng.integers(0, 2, n_artefact)]
    art_f = rng.uniform(*artefact_vaf, n_artefact)

    n_reads = length * depth // read_length
    starts = np.sort(rng.integers(0, length - read_length, n_reads))
    pos = starts[:, None] + np.arange(read_length)
    true_seq = np.where(rng.integers(0, 2, n_reads)[:, None] == 1, hap1[pos], ref[pos])

    # Each base errs to dst with probability m[src, dst]: one uniform per base,
    # partitioned into consecutive intervals.
    u = rng.random(true_seq.shape)
    seq = true_seq.copy()
    for s in range(4):
        is_s = true_seq == s
        lo = 0.0
        for d in range(4):
            if d != s:
                seq[is_s & (u >= lo) & (u < lo + m[s, d])] = d
                lo += m[s, d]
    for p, x, f in zip(art, art_x, art_f):
        seq[(pos == p) & (rng.random(seq.shape) < f)] = x

    sam = outdir / "reads.sam"
    qual = "?" * read_length  # Q30
    with open(sam, "w") as fh:
        fh.write(f"@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:{length}\n")
        for i in range(n_reads):
            fh.write(
                f"r{i}\t0\tchr1\t{starts[i] + 1}\t60\t{read_length}M\t*\t0\t0\t"
                f"{''.join(BASES[seq[i]])}\t{qual}\n"
            )
    return ReadSimulation(fasta, sam, true_rates(rate, profile), het + 1, art + 1)
