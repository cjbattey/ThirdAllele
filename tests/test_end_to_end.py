"""Full pipeline on simulated reads: bcftools mpileup/call -> estimator -> CLI outputs."""

from __future__ import annotations

import shutil
import subprocess
import sys

import polars as pl
import pytest

from third_allele import MUTATION_TYPES, ThirdAlleleErrorEstimator
from third_allele.simulate import ReadSimulation, simulate_reads

pytestmark = [
    pytest.mark.bcftools,
    pytest.mark.skipif(shutil.which("bcftools") is None, reason="bcftools not on PATH"),
]


@pytest.fixture(scope="module")
def sim(tmp_path_factory) -> ReadSimulation:
    return simulate_reads(
        tmp_path_factory.mktemp("reads"),
        length=150_000,
        depth=200,
        n_het=800,
        n_artefact=15,
        rate=2e-3,
        seed=0,
    )


@pytest.fixture(scope="module")
def cli_run(sim, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    proc = subprocess.run(
        [
            sys.executable, "-m", "third_allele",
            str(sim.reads), str(sim.reference),
            "-o", str(out / "errors.tsv"),
            "-s", str(out / "sites.tsv"),
        ],
        capture_output=True,
        text=True,
    )  # fmt: skip
    return proc, out


def test_cli_succeeds(cli_run):
    proc, out = cli_run
    assert proc.returncode == 0, proc.stderr
    assert "overall" in proc.stdout
    assert (out / "errors.tsv").exists() and (out / "sites.tsv").exists()


def test_all_het_sites_called(sim, cli_run):
    sites = pl.read_csv(cli_run[1] / "sites.tsv", separator="\t")
    assert set(sites["pos"]) == set(sim.het_positions.tolist())
    assert sites["used"].all()


def test_error_table_matches_truth(sim, cli_run):
    tab = pl.read_csv(cli_run[1] / "errors.tsv", separator="\t")
    assert tab["mutation_type"].to_list() == MUTATION_TYPES + ["overall"]
    rows = {r["mutation_type"]: r for r in tab.iter_rows(named=True)}
    overall = rows["overall"]
    assert overall["ci_low"] <= sim.truth["overall"] <= overall["ci_high"]
    assert overall["error_rate"] == pytest.approx(sim.truth["overall"], rel=0.1)
    # With only ~800 sites the per-class split between the two possible source
    # bases is noisy (see test_simulation.py for per-class accuracy at scale),
    # so check internal consistency and the dominant class only.
    assert sum(rows[t]["errors"] for t in MUTATION_TYPES) == pytest.approx(overall["errors"])
    assert max(MUTATION_TYPES, key=lambda t: rows[t]["error_rate"]) == "C>T"
    assert (tab["lod_depth"] == 200).all()


def test_artefact_sites_flagged(sim, cli_run):
    sites = pl.read_csv(cli_run[1] / "sites.tsv", separator="\t")
    flagged = set(sites.filter(pl.col("artefact_weight") > 0.5)["pos"])
    assert flagged == set(sim.artefact_positions.tolist())


def test_no_artefact_model_counts_artefacts_as_errors(sim):
    est = ThirdAlleleErrorEstimator(bam=sim.reads, reference=sim.reference, artefact_model=False)
    rate = est.error_table()["error_rate"][-1]
    assert rate > 1.2 * sim.truth["overall"]


def test_bcftools_failure_is_reported(sim, tmp_path):
    est = ThirdAlleleErrorEstimator(bam=tmp_path / "missing.bam", reference=sim.reference)
    with pytest.raises(RuntimeError, match="mpileup"):
        est.raw_sites
