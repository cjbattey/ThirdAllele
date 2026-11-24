"""Estimate sequencing error rates from "third alleles" at germline heterozygous SNPs.

At a germline het site (REF/ALT) every read should carry either REF or ALT. A
read carrying a third base is either a sequencing error or comes from an
artefact: a mapping problem, an unresolved multi-allelic locus, or
contamination.

The two are separated with a mixture model fitted by EM. For each site and each
of its two possible third alleles x, the number of x reads comes from one of:

* Errors: Poisson(lambda), where lambda = sum over the germline alleles S of
  reads(S) * rate(S>x).
* An artefact: Binomial(depth, f), with f ~ Uniform(f_min, artefact_max_vaf).

Each (site, x) gets a posterior artefact weight w. It contributes (1 - w) of
its x reads to the error counts and (1 - w) of its germline reads to the
denominators. Nothing is discarded outright, so there's no hard truncation.

The artefact floor f_min is max(artefact_min_reads, Q) / depth, where Q is the
``artefact_tail_quantile`` quantile of the error count Poisson(lambda). That
keeps the artefact component clear of the bulk of the error distribution
whether lambda is tiny (duplex consensus data, low-depth WGS) or several reads
(raw high-depth capture). Floors depend on the rates, so they are recomputed
between EM rounds until they stop changing. Fitting starts from low rates
(the one-error-per-site estimate, or a user prior) so the floors climb from
below rather than settling where artefacts are treated as errors.

Because the source allele of an error read is unknown, the error part of each
third-allele count is split between the two germline alleles in proportion to
``reads(source) * rate(source>third)``, refined by the same EM.

Substitutions are folded onto the pyrimidine strand of the source base, so the
six classes are C>A, C>G, C>T, T>A, T>C and T>G (e.g. G>A is counted as C>T).

Denominators are per-allele read counts, and they skip substitutions that can't
be seen: at an A/T het site the A reads are opportunities for A>C and A>G only
(an A>T error looks like a germline T read), and the T reads are opportunities
for T>C and T>G only.

Requires bcftools on PATH, plus polars, numpy and scipy.
"""

from __future__ import annotations

import io
import logging
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from scipy.optimize import brentq
from scipy.special import gammaln
from scipy.stats import beta, binom, poisson

log = logging.getLogger("third_allele_errors")

BASES = ["A", "C", "G", "T"]
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}
PYRIMIDINE_REFS = ["C", "T"]
MUTATION_TYPES = [f"{r}>{a}" for r in PYRIMIDINE_REFS for a in BASES if a != r]
N_TYPES = len(MUTATION_TYPES)
QUERY_COLUMNS = ["chrom", "pos", "ref", "alt", "qual", "gt", "ad"]
SITE_KEY = ["chrom", "pos"]
EM_TOL = 1e-6
MAX_FLOOR_ROUNDS = 50
INITIAL_PI = 0.01
# Beta(1.5, 1.5) prior on the artefact fraction: keeps it off exactly 0 or 1,
# where EM would get stuck, and is negligible with more than a few sites.
PI_PSEUDOCOUNT = 0.5
TINY = 1e-300
LOG_PMF_FLOOR = -36.0  # ~2e-16; Poisson terms below this are ignored


def _fold(col: str, src_col: str) -> pl.Expr:
    """Complement ``col`` if the source base ``src_col`` is a purine (A/G)."""
    return (
        pl.when(pl.col(src_col).is_in(["A", "G"]))
        .then(pl.col(col).replace(COMPLEMENT))
        .otherwise(pl.col(col))
    )


def _mutation_type(src: str, dst: str) -> pl.Expr:
    """Pyrimidine-folded substitution label, e.g. G>A -> 'C>T'."""
    return pl.concat_str([_fold(src, src), pl.lit(">"), _fold(dst, src)])


def clopper_pearson(k: float, n: float, conf: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval for k successes out of n trials.

    ``k`` and ``n`` may be fractional (from weighted counts); the beta form
    handles it.
    """
    if n <= 0:
        return math.nan, math.nan
    a = 1 - conf
    lo = 0.0 if k <= 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k >= n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def truncation_loss(mean_kept: float, max_errors: int) -> float:
    """Fraction of error reads lost by dropping sites with > ``max_errors`` errors.

    Assumes errors per site ~ Poisson(lam). Kept sites have mean
    E[N | N <= k] = lam * P(N <= k-1) / P(N <= k); solve that for lam, then the
    lost fraction of error reads is 1 - P(N <= k-1). This ignores variation in
    depth between sites, so treat it as a rough diagnostic.
    """
    k = max_errors
    if mean_kept <= 0 or k <= 0:
        return 0.0

    def kept_mean_minus_obs(lam: float) -> float:
        log_ratio = poisson.logcdf(k - 1, lam) - poisson.logcdf(k, lam)
        return lam * math.exp(log_ratio) - mean_kept

    hi = 100.0 * (k + 1)
    if mean_kept >= k or kept_mean_minus_obs(hi) < 0:
        return math.nan
    lam = brentq(kept_mean_minus_obs, 1e-12, hi)
    return float(1 - poisson.cdf(k - 1, lam))


def limit_of_detection(
    error_rate: float, depth: int, alpha: float = 0.05, power: float = 0.95
) -> tuple[int, float]:
    """LoB/LoD in the style of CLSI EP17, using a binomial error model.

    1. Limit of blank: the minimum alt-read count ``k`` such that a site with
       no true variant (errors only, ~ Binomial(depth, error_rate)) reaches
       ``k`` reads with probability <= ``alpha``.
    2. Limit of detection: the smallest VAF at which a true variant yields
       at least ``k`` alt reads with probability >= ``power``.

    Returns ``(min_alt_reads, lod_vaf)``. ``lod_vaf`` is NaN if ``k`` > ``depth``.
    """
    if depth <= 0 or math.isnan(error_rate):
        return 0, math.nan
    # smallest k with P(X >= k) <= alpha  <=>  k = ppf(1 - alpha) + 1
    k = int(binom.ppf(1 - alpha, depth, error_rate)) + 1
    if k > depth:
        return k, math.nan

    def detect_prob_minus_power(vaf: float) -> float:
        return float(binom.sf(k - 1, depth, vaf)) - power

    # P(X >= k) increases monotonically with VAF and reaches 1 at VAF=1.
    lod = brentq(detect_prob_minus_power, 1e-12, 1.0, xtol=1e-10)
    return k, float(lod)


# ---------------------------------------------------------------------- mixture


def log_poisson(c, lam) -> np.ndarray:
    """log Poisson(c; lam), with lam floored at a tiny positive value."""
    lam = np.maximum(lam, TINY)
    return c * np.log(lam) - lam - gammaln(np.asarray(c, dtype=float) + 1)


def log_artefact_likelihood(c, depth, lo, hi) -> np.ndarray:
    """log P(c | artefact): Binomial(depth, f) averaged over f ~ Uniform(lo, hi).

    Uses  integral_lo^hi Binom(c; n, f) df = (I(hi) - I(lo)) / (n + 1),  where I
    is the Beta(c + 1, n - c + 1) CDF. The difference is taken on whichever tail
    avoids cancellation. Returns -inf where lo >= hi.
    """
    c, depth, lo, hi = (
        np.array(x, dtype=float) for x in np.broadcast_arrays(c, depth, lo, hi)
    )
    a = c + 1.0
    b = depth - c + 1.0
    upper = lo > a / (a + b)  # whole interval above the mode: use survival fns
    mass = np.empty_like(c)
    m = upper
    mass[m] = beta.sf(lo[m], a[m], b[m]) - beta.sf(hi[m], a[m], b[m])
    m = ~upper
    mass[m] = beta.cdf(hi[m], a[m], b[m]) - beta.cdf(lo[m], a[m], b[m])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(np.clip(mass, 0.0, None)) - np.log(hi - lo) - np.log(depth + 1)
    return np.where(hi > lo, out, -np.inf)


def artefact_posterior(log_err, log_art, pi: float) -> np.ndarray:
    """P(artefact | count) for a two-component mixture with prior weight ``pi``."""
    if pi <= 0:
        return np.zeros_like(log_err, dtype=float)
    la = math.log(pi) + log_art
    le = math.log1p(-pi) + log_err
    return np.exp(la - np.logaddexp(la, le))


def artefact_floor_reads(lam, min_reads: int, quantile: float) -> np.ndarray:
    """Read-count floor for the artefact component: max(min_reads, Q_quantile(lam))."""
    return np.maximum(min_reads, poisson.ppf(quantile, np.maximum(lam, TINY)))


def _rel_change(new, old) -> float:
    new, old = np.asarray(new, dtype=float), np.asarray(old, dtype=float)
    if new.size == 0:
        return 0.0
    return float(np.max(np.abs(new - old) / np.maximum(np.abs(old), 1e-12)))


@dataclass
class SiteAlleleArrays:
    """Arrays for the EM, one entry per (used site, possible third allele x)."""

    count: np.ndarray  # reads carrying x
    depth: np.ndarray  # site depth (all four bases)
    n_ref: np.ndarray  # reads carrying the ref allele
    n_alt: np.ndarray  # reads carrying the germline alt allele
    type_ref: np.ndarray  # class index of ref>x
    type_alt: np.ndarray  # class index of alt>x
    site_errors: np.ndarray  # third-allele reads at the site (both x)

    def expected_errors(self, rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Expected error reads of x arising from the ref and from the alt reads."""
        return self.n_ref * rates[self.type_ref], self.n_alt * rates[self.type_alt]

    def e_step(
        self, rates: np.ndarray, pi: float, log_art: Optional[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Artefact weight per row, and the share of its error reads from ref."""
        lam_ref, lam_alt = self.expected_errors(rates)
        lam = lam_ref + lam_alt
        if log_art is None:
            w = np.zeros_like(lam)
        else:
            w = artefact_posterior(log_poisson(self.count, lam), log_art, pi)
        share_ref = self.n_ref / (self.n_ref + self.n_alt)  # fallback: read split
        np.divide(lam_ref, lam, out=share_ref, where=lam > 0)
        return w, share_ref

    def m_step(
        self, w: np.ndarray, share_ref: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Weighted error reads, opportunities and rates per class."""
        keep = 1.0 - w
        err = keep * self.count
        errors = np.bincount(
            self.type_ref, weights=err * share_ref, minlength=N_TYPES
        ) + np.bincount(self.type_alt, weights=err * (1 - share_ref), minlength=N_TYPES)
        opps = np.bincount(
            self.type_ref, weights=keep * self.n_ref, minlength=N_TYPES
        ) + np.bincount(self.type_alt, weights=keep * self.n_alt, minlength=N_TYPES)
        rates = np.divide(errors, opps, out=np.zeros(N_TYPES), where=opps > 0)
        return errors, opps, rates


@dataclass
class MixtureFit:
    """Fitted mixture. Per-row arrays line up with ``site_alleles``."""

    rates: np.ndarray  # per class
    pi: float  # prior artefact probability per (site, third allele)
    weight: np.ndarray  # posterior artefact weight per row
    errors: np.ndarray  # weighted error reads per class
    opportunities: np.ndarray  # weighted opportunities per class
    floor_reads: Optional[np.ndarray]  # artefact floor per row (None: no model)


@dataclass
class ThirdAlleleErrorEstimator:
    """Estimate per-substitution error rates from third alleles at germline het SNPs.

    Attributes:
        bam: Indexed, coordinate-sorted BAM/CRAM for a single sample.
        reference: Indexed reference FASTA the reads were aligned to.
        regions: Optional region string (``chr1:1-1000``) or BED file to restrict pileup.
        min_mapq: Minimum mapping quality for a read to be counted.
        min_baseq: Minimum base quality for a base to be counted.
        max_depth: Max per-file depth for ``bcftools mpileup`` (its default of
            250 is too low for most error-rate work).
        min_qual: Minimum variant QUAL for a het site to be used.
        min_depth: Minimum depth (sum of AD over A/C/G/T) at a het site.
        max_site_depth: Maximum depth at a het site (removes collapsed repeats).
        min_het_vaf: Minimum germline ALT/(REF+ALT) for a site to count as het.
        max_het_vaf: Maximum germline ALT/(REF+ALT) for a site to count as het.
        max_errors_per_site: Optional hard cap: sites with more third-allele
            reads than this are discarded before fitting. This biases rates low
            (more so at high depth); the artefact model makes it unnecessary.
        artefact_model: Fit the error/artefact mixture. If False, every
            third-allele read is counted as an error.
        artefact_min_reads: Minimum read-count floor of the artefact component.
        artefact_tail_quantile: The artefact floor is at least this quantile of
            the expected error count, keeping it clear of genuine errors.
        artefact_max_vaf: Upper bound of the artefact VAF prior.
        prior_error_rate: Starting per-substitution error rate for the fit.
            Default: the one-error-per-site estimate, which errs low.
        max_em_iterations: Maximum EM iterations per artefact-floor round.
        threads: Extra compression threads passed to bcftools.
    """

    bam: Path
    reference: Path
    regions: Optional[str] = None
    min_mapq: int = 20
    min_baseq: int = 20
    max_depth: int = 10_000
    min_qual: float = 30.0
    min_depth: int = 20
    max_site_depth: Optional[int] = None
    min_het_vaf: float = 0.3
    max_het_vaf: float = 0.7
    max_errors_per_site: Optional[int] = None
    artefact_model: bool = True
    artefact_min_reads: int = 3
    artefact_tail_quantile: float = 0.999
    artefact_max_vaf: float = 0.5
    prior_error_rate: Optional[float] = None
    max_em_iterations: int = 1000
    threads: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.artefact_tail_quantile < 1:
            raise ValueError("artefact_tail_quantile must be in (0, 1)")
        if not 0 < self.artefact_max_vaf <= 1:
            raise ValueError("artefact_max_vaf must be in (0, 1]")
        if self.prior_error_rate is not None and not 0 < self.prior_error_rate < 1:
            raise ValueError("prior_error_rate must be in (0, 1)")
        if self.max_em_iterations < 1:
            raise ValueError("max_em_iterations must be at least 1")

    # ------------------------------------------------------------------ bcftools

    def _bcftools_commands(self) -> list[list[str]]:
        threads = ["--threads", str(self.threads)] if self.threads else []
        region_args: list[str] = []
        if self.regions:
            flag = "-R" if Path(self.regions).is_file() else "-r"
            region_args = [flag, self.regions]

        mpileup = [
            "bcftools", "mpileup",
            "-f", str(self.reference),
            "-a", "FORMAT/AD,FORMAT/DP",
            "-q", str(self.min_mapq),
            "-Q", str(self.min_baseq),
            "-d", str(self.max_depth),
            "--ignore-RG",
            *region_args, *threads,
            "-Ou", str(self.bam),
        ]  # fmt: skip
        # -A keeps every observed allele (and its AD) even if it isn't in the
        # called genotype. Without it, bcftools call trims the third allele.
        call = ["bcftools", "call", "-m", "-v", "-A", *threads, "-Ou"]
        view = [
            "bcftools", "view",
            "-v", "snps",
            "-g", "het",
            "-i", f"QUAL>={self.min_qual}",
            *threads, "-Ou",
        ]  # fmt: skip
        query = [
            "bcftools", "query",
            "-f", "%CHROM\t%POS\t%REF\t%ALT\t%QUAL[\t%GT\t%AD]\n",
        ]  # fmt: skip
        return [mpileup, call, view, query]

    def _run_pipeline(self) -> str:
        if shutil.which("bcftools") is None:
            raise RuntimeError("bcftools not found on PATH")
        cmds = self._bcftools_commands()
        log.info("Running: %s", " | ".join(" ".join(c) for c in cmds))

        # stderr goes to temp files so a chatty process can't fill a pipe and stall.
        errs = [tempfile.TemporaryFile() for _ in cmds]
        procs: list[subprocess.Popen] = []
        stdin = None
        for cmd, err in zip(cmds, errs):
            p = subprocess.Popen(cmd, stdin=stdin, stdout=subprocess.PIPE, stderr=err)
            if stdin is not None:
                stdin.close()  # let upstream get SIGPIPE if downstream exits
            stdin = p.stdout
            procs.append(p)

        out, _ = procs[-1].communicate()
        failed = []
        for cmd, p, err in zip(cmds, procs, errs):
            p.wait()
            err.seek(0)
            msg = err.read().decode().strip()
            err.close()
            if p.returncode != 0:
                failed.append(f"{cmd[1]} (exit {p.returncode}): {msg}")
        if failed:
            raise RuntimeError("bcftools pipeline failed:\n" + "\n".join(failed))
        return out.decode()

    # ------------------------------------------------------------------ tables

    @cached_property
    def raw_sites(self) -> pl.DataFrame:
        """One row per het SNP record emitted by bcftools."""
        text = self._run_pipeline()
        schema = {c: pl.Utf8 for c in QUERY_COLUMNS}
        schema["pos"] = pl.Int64
        if not text.strip():
            return pl.DataFrame(schema=schema)
        return pl.read_csv(
            io.StringIO(text),
            separator="\t",
            has_header=False,
            schema=schema,
        )

    @cached_property
    def allele_counts(self) -> pl.DataFrame:
        """Long table: one row per (site, observed base) with its read count and role.

        ``role`` is ``ref``, ``alt`` (germline het allele) or ``err`` (third allele).
        Only ``0/x`` genotypes are kept, so the reference is always one of the
        germline alleles.
        """
        df = self.raw_sites.with_columns(
            alleles=pl.concat_list(pl.col("ref"), pl.col("alt").str.split(",")),
            ad=pl.col("ad").str.split(",").cast(pl.List(pl.Int64)),
            gt_idx=pl.col("gt").str.extract_all(r"\d+").cast(pl.List(pl.Int64)),
        )
        df = (
            df.filter(
                (pl.col("gt_idx").list.len() == 2)
                & (pl.col("gt_idx").list.min() == 0)
                & (pl.col("gt_idx").list.max() > 0)
                & (pl.col("alleles").list.len() == pl.col("ad").list.len())
            )
            .with_columns(
                germ_alt=pl.col("alleles").list.get(pl.col("gt_idx").list.max())
            )
            .filter(pl.col("ref").is_in(BASES) & pl.col("germ_alt").is_in(BASES))
        )
        return (
            df.select("chrom", "pos", "ref", "germ_alt", "alleles", "ad")
            .explode(["alleles", "ad"])
            .rename({"alleles": "allele"})
            .filter(pl.col("allele").is_in(BASES))  # drop <*> etc.
            .with_columns(
                role=pl.when(pl.col("allele") == pl.col("ref"))
                .then(pl.lit("ref"))
                .when(pl.col("allele") == pl.col("germ_alt"))
                .then(pl.lit("alt"))
                .otherwise(pl.lit("err"))
            )
        )

    @cached_property
    def sites(self) -> pl.DataFrame:
        """Per-site summary with QC flags. ``used`` marks sites in the estimate."""
        role = pl.col("role")
        ad = pl.col("ad")
        sites = (
            self.allele_counts.group_by("chrom", "pos", "ref", "germ_alt")
            .agg(
                depth=ad.sum(),
                ref_count=ad.filter(role == "ref").sum(),
                alt_count=ad.filter(role == "alt").sum(),
                n_err=ad.filter(role == "err").sum(),
            )
            .with_columns(
                het_vaf=pl.col("alt_count")
                / (pl.col("ref_count") + pl.col("alt_count"))
            )
            .sort("chrom", "pos")
        )
        depth_ok = pl.col("depth") >= self.min_depth
        if self.max_site_depth is not None:
            depth_ok = depth_ok & (pl.col("depth") <= self.max_site_depth)
        het_ok = pl.col("het_vaf").is_between(self.min_het_vaf, self.max_het_vaf)
        used = depth_ok & het_ok
        if self.max_errors_per_site is not None:
            used = used & (pl.col("n_err") <= self.max_errors_per_site)
        return sites.with_columns(pass_qc=depth_ok & het_ok, used=used)

    @cached_property
    def used_sites(self) -> pl.DataFrame:
        return self.sites.filter("used")

    @cached_property
    def site_alleles(self) -> pl.DataFrame:
        """One row per (used site, possible third allele x), including unseen x.

        ``count`` is the number of x reads (0 if none were seen). ``type_ref``
        and ``type_alt`` are the folded classes ref>x and alt>x. Rows with no x
        reads still matter: their germline reads are opportunities for errors.
        """
        third = self.allele_counts.filter(pl.col("role") == "err").select(
            *SITE_KEY, x=pl.col("allele"), count=pl.col("ad")
        )
        return (
            self.used_sites.select(
                *SITE_KEY, "ref", "germ_alt", "depth", "ref_count", "alt_count", "n_err"
            )
            .join(pl.DataFrame({"x": BASES}), how="cross")
            .filter((pl.col("x") != pl.col("ref")) & (pl.col("x") != pl.col("germ_alt")))
            .join(third, on=[*SITE_KEY, "x"], how="left")
            .with_columns(
                pl.col("count").fill_null(0),
                type_ref=_mutation_type("ref", "x"),
                type_alt=_mutation_type("germ_alt", "x"),
            )
            .sort(*SITE_KEY, "x")
        )

    @cached_property
    def arrays(self) -> SiteAlleleArrays:
        sa = self.site_alleles

        def num(col: str) -> np.ndarray:
            return sa[col].cast(pl.Float64).to_numpy()

        def type_index(col: str) -> np.ndarray:
            return (
                sa[col].cast(pl.Enum(MUTATION_TYPES)).to_physical().to_numpy()
            ).astype(np.intp)

        return SiteAlleleArrays(
            count=num("count"),
            depth=num("depth"),
            n_ref=num("ref_count"),
            n_alt=num("alt_count"),
            type_ref=type_index("type_ref"),
            type_alt=type_index("type_alt"),
            site_errors=num("n_err"),
        )

    # ------------------------------------------------------------------ fitting

    def _initial_rates(self, d: SiteAlleleArrays) -> np.ndarray:
        if self.prior_error_rate is not None:
            return np.full(N_TYPES, self.prior_error_rate)
        # The one-error-per-site estimate with a read-count split. It's biased
        # low, so the artefact floors start low and rise as the rates do.
        w0 = (d.site_errors > 1).astype(float)
        return d.m_step(w0, d.n_ref / (d.n_ref + d.n_alt))[2]

    def _artefact_floor(self, d: SiteAlleleArrays, rates: np.ndarray) -> np.ndarray:
        lam_ref, lam_alt = d.expected_errors(rates)
        return artefact_floor_reads(
            lam_ref + lam_alt, self.artefact_min_reads, self.artefact_tail_quantile
        )

    def _log_artefact(self, d: SiteAlleleArrays, floor: np.ndarray) -> np.ndarray:
        return log_artefact_likelihood(
            d.count, d.depth, floor / d.depth, self.artefact_max_vaf
        )

    def _em(
        self,
        d: SiteAlleleArrays,
        rates: np.ndarray,
        pi: float,
        log_art: Optional[np.ndarray],
    ) -> tuple[np.ndarray, float]:
        """EM over the artefact weights, source split, rates and pi."""
        for i in range(1, self.max_em_iterations + 1):
            w, share_ref = d.e_step(rates, pi, log_art)
            new_rates = d.m_step(w, share_ref)[2]
            new_pi = (
                (w.sum() + PI_PSEUDOCOUNT) / (w.size + 2 * PI_PSEUDOCOUNT)
                if log_art is not None
                else 0.0
            )
            delta = max(_rel_change(new_rates, rates), _rel_change(new_pi, pi))
            rates, pi = new_rates, float(new_pi)
            if delta < EM_TOL:
                log.debug("EM converged after %d iterations", i)
                return rates, pi
        log.warning("EM did not converge in %d iterations", self.max_em_iterations)
        return rates, pi

    @cached_property
    def fit(self) -> MixtureFit:
        """Fit the error/artefact mixture (or the error-only model).

        Alternates between setting the artefact floors from the current rates
        and running EM with those floors fixed, until the floors stop changing.
        """
        d = self.arrays
        rates = self._initial_rates(d)
        pi = 0.0
        log_art: Optional[np.ndarray] = None
        floor: Optional[np.ndarray] = None
        if not self.artefact_model:
            rates, pi = self._em(d, rates, 0.0, None)
        else:
            pi = INITIAL_PI
            for rnd in range(1, MAX_FLOOR_ROUNDS + 1):
                new_floor = self._artefact_floor(d, rates)
                if floor is not None and np.array_equal(new_floor, floor):
                    break
                floor = new_floor
                log_art = self._log_artefact(d, floor)
                rates, pi = self._em(d, rates, pi, log_art)
                log.info(
                    "Artefact round %d: pi=%.3g, mean rate=%.3g",
                    rnd, pi, float(rates.mean()),
                )  # fmt: skip
            else:
                log.warning(
                    "Artefact floors still changing after %d rounds", MAX_FLOOR_ROUNDS
                )
        w, share_ref = d.e_step(rates, pi, log_art)
        errors, opps, _ = d.m_step(w, share_ref)
        return MixtureFit(
            rates=rates,
            pi=pi,
            weight=w,
            errors=errors,
            opportunities=opps,
            floor_reads=floor,
        )

    @cached_property
    def rate_bias(self) -> float:
        """Expected relative bias of the pooled rate from soft-weighting real errors.

        Treats every (site, x) as error-only with the fitted lambda and asks how
        much the artefact weights would remove from its error reads and from its
        opportunities. Negative means rates are underestimated. This is the
        mixture's counterpart to ``truncation_loss``.
        """
        f, d = self.fit, self.arrays
        if f.floor_reads is None or f.pi <= 0 or d.count.size == 0:
            return 0.0
        lam = sum(d.expected_errors(f.rates))
        n = d.n_ref + d.n_alt
        if lam.sum() <= 0:
            return 0.0
        lo = f.floor_reads / d.depth
        lost_reads = lost_opps = 0.0
        c_max = min(int(poisson.ppf(1 - 1e-12, lam.max())) + 1, 10_000)
        for c in range(c_max + 1):
            log_p = log_poisson(c, lam)
            m = log_p > LOG_PMF_FLOOR
            if not m.any():
                continue
            la = log_artefact_likelihood(c, d.depth[m], lo[m], self.artefact_max_vaf)
            pw = np.exp(log_p[m]) * artefact_posterior(log_p[m], la, f.pi)
            lost_reads += c * pw.sum()
            lost_opps += (n[m] * pw).sum()
        return float((1 - lost_reads / lam.sum()) / (1 - lost_opps / n.sum()) - 1)

    @cached_property
    def median_depth(self) -> int:
        if self.used_sites.height == 0:
            return 0
        return int(self.used_sites["depth"].median())

    def sites_report(self) -> pl.DataFrame:
        """``sites`` plus, for used sites, fitted artefact reads and max weight."""
        per_site = (
            self.site_alleles.select(*SITE_KEY, "count")
            .with_columns(pl.Series("w", self.fit.weight))
            .group_by(SITE_KEY)
            .agg(
                artefact_reads=(pl.col("w") * pl.col("count")).sum(),
                artefact_weight=pl.col("w").max(),
            )
        )
        return self.sites.join(per_site, on=SITE_KEY, how="left").sort(SITE_KEY)

    def error_table(
        self,
        lod_depth: Optional[int] = None,
        alpha: float = 0.05,
        power: float = 0.95,
    ) -> pl.DataFrame:
        """Error rate per folded substitution class plus an overall row.

        Columns:
            errors: third-allele reads assigned to this class, weighted by
                1 - artefact weight (fractional).
            opportunities: reads carrying a source base where this class is
                observable, weighted the same way.
            error_rate, ci_low, ci_high: rate and exact 95% CI.
            lob_alt_reads: alt reads needed to call a variant at ``lod_depth``
                with false-positive rate <= ``alpha``.
            lod_vaf: VAF detected with probability >= ``power`` at that threshold.

        The ``overall`` row pools all six classes, so its rate is the mean
        per-substitution error rate, directly comparable with the class rows.
        A per-base rate (any substitution) is about 3x this.
        """
        depth = lod_depth if lod_depth is not None else self.median_depth
        f = self.fit
        table = pl.DataFrame(
            {
                "mutation_type": MUTATION_TYPES + ["overall"],
                "errors": [*f.errors, f.errors.sum()],
                "opportunities": [*f.opportunities, f.opportunities.sum()],
            },
            schema={
                "mutation_type": pl.Utf8,
                "errors": pl.Float64,
                "opportunities": pl.Float64,
            },
        )
        rows = []
        for k, n in table.select("errors", "opportunities").iter_rows():
            rate = k / n if n else math.nan
            lo, hi = clopper_pearson(k, n)
            lob, lod = limit_of_detection(rate, depth, alpha, power)
            rows.append((rate, lo, hi, lob, lod))
        stats = pl.DataFrame(
            rows,
            schema=["error_rate", "ci_low", "ci_high", "lob_alt_reads", "lod_vaf"],
            orient="row",
        )
        return pl.concat([table, stats], how="horizontal").with_columns(
            lod_depth=pl.lit(depth, dtype=pl.Int64)
        )

    def summary(self) -> dict[str, float]:
        s = self.sites
        f = self.fit
        count = self.arrays.count
        report = self.sites_report()
        return {
            "het_records": self.raw_sites.height,
            "het_0x_snps": s.height,
            "pass_qc": int(s["pass_qc"].sum()),
            "dropped_by_cap": int((s["pass_qc"] & ~s["used"]).sum()),
            "used": int(s["used"].sum()),
            "used_with_error": int((s["used"] & (s["n_err"] > 0)).sum()),
            "third_allele_reads": int(count.sum()),
            "artefact_reads": float((f.weight * count).sum()),
            "artefact_sites": int((report["artefact_weight"] > 0.5).sum()),
            "artefact_fraction": f.pi,
            "median_depth": self.median_depth,
        }
