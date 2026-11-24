from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import pytest

from third_allele import ThirdAlleleErrorEstimator


def make_estimator(raw_sites: pl.DataFrame, **kwargs) -> ThirdAlleleErrorEstimator:
    """Estimator fed a pre-built ``bcftools query`` table, bypassing bcftools."""
    est = ThirdAlleleErrorEstimator(bam=Path("unused.bam"), reference=Path("unused.fa"), **kwargs)
    est.raw_sites = raw_sites  # overrides the cached_property
    return est


@pytest.fixture
def estimator():
    return make_estimator


@pytest.fixture(autouse=True)
def _quiet_logs(caplog):
    caplog.set_level(logging.WARNING, logger="third_allele_errors")
