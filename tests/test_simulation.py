"""Recover known error rates from simulated het sites, with and without artefacts.

Tolerances were set from runs over several seeds. The overall rate is tight
everywhere. Per-class rates are looser because the EM has to split each
third-allele read between its two possible source bases, which adds noise
beyond the Poisson count (most for the rare classes at low depth).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from third_allele import MUTATION_TYPES
from third_allele.simulate import SiteSimulation, simulate_sites

from conftest import make_estimator


@dataclass(frozen=True)
class Regime:
    n: int
    depth: float
    rate: float
    artefact_fraction: float
    class_tol: float  # allowed relative error per class


REGIMES = {
    "wgs_30x": Regime(n=200_000, depth=30, rate=1e-3, artefact_fraction=0.01, class_tol=0.30),
    "capture_2000x": Regime(n=20_000, depth=2000, rate=1e-3, artefact_fraction=0.02, class_tol=0.15),
    "duplex_2000x": Regime(n=200_000, depth=2000, rate=5e-7, artefact_fraction=0.005, class_tol=0.35),
}
CONFIGS = {
    "mixture": {},
    "no_filter": dict(artefact_model=False),
    "legacy_cap": dict(artefact_model=False, max_errors_per_site=1),
}
OVERALL_TOL = 0.10

_cache: dict = {}


def simulation(regime: str) -> SiteSimulation:
    if regime not in _cache:
        r = REGIMES[regime]
        _cache[regime] = simulate_sites(
            n=r.n, depth=r.depth, rate=r.rate, artefact_fraction=r.artefact_fraction, seed=0
        )
    return _cache[regime]


def fitted(regime: str, config: str, **extra):
    key = (regime, config, tuple(sorted(extra.items())))
    if key not in _cache:
        est = make_estimator(simulation(regime).raw_sites, **CONFIGS[config], **extra)
        tab = est.error_table()
        _cache[key] = est, dict(zip(tab["mutation_type"], tab["error_rate"]))
    return _cache[key]


def ratio(regime: str, config: str, mutation_type: str = "overall") -> float:
    return fitted(regime, config)[1][mutation_type] / simulation(regime).truth[mutation_type]


regimes = pytest.mark.parametrize(
    "regime", [pytest.param(k, marks=pytest.mark.slow) if REGIMES[k].n > 50_000 else k for k in REGIMES]
)


@regimes
def test_mixture_recovers_overall_rate(regime):
    assert ratio(regime, "mixture") == pytest.approx(1, abs=OVERALL_TOL)


@regimes
@pytest.mark.parametrize("mutation_type", MUTATION_TYPES)
def test_mixture_recovers_each_class(regime, mutation_type):
    tol = REGIMES[regime].class_tol
    assert ratio(regime, "mixture", mutation_type) == pytest.approx(1, abs=tol)


@regimes
def test_mixture_recovers_spectrum_shape(regime):
    """Highest and lowest classes are identified, and rates are well correlated."""
    rates = fitted(regime, "mixture")[1]
    truth = simulation(regime).truth
    est = np.array([rates[t] for t in MUTATION_TYPES])
    true = np.array([truth[t] for t in MUTATION_TYPES])
    assert MUTATION_TYPES[est.argmax()] == MUTATION_TYPES[true.argmax()]
    assert MUTATION_TYPES[est.argmin()] == MUTATION_TYPES[true.argmin()]
    assert np.corrcoef(np.log(est), np.log(true))[0, 1] > 0.95


@regimes
def test_no_filter_is_inflated_by_artefacts(regime):
    """Without the model, artefact reads count as errors: the reason it exists."""
    assert ratio(regime, "no_filter") > 1.4


def test_legacy_cap_is_biased_low_at_high_depth():
    assert ratio("capture_2000x", "legacy_cap") < 0.3


@regimes
def test_prior_rate_converges_to_same_answer(regime):
    est, rates = fitted(regime, "mixture", prior_error_rate=REGIMES[regime].rate)
    assert rates["overall"] == pytest.approx(fitted(regime, "mixture")[1]["overall"], rel=0.01)


@regimes
def test_estimated_rate_bias_is_small(regime):
    est, _ = fitted(regime, "mixture")
    assert abs(est.rate_bias) < 0.02


@pytest.mark.parametrize(
    "regime", ["capture_2000x", pytest.param("duplex_2000x", marks=pytest.mark.slow)]
)
def test_artefact_sites_identified_at_high_depth(regime):
    est, _ = fitted(regime, "mixture")
    sim = simulation(regime)
    flagged = est.sites_report().sort("pos")["artefact_weight"].to_numpy() > 0.5
    assert flagged.size == sim.is_artefact.size
    recall = (flagged & sim.is_artefact).sum() / sim.is_artefact.sum()
    false_pos = (flagged & ~sim.is_artefact).sum()
    assert recall > 0.95
    assert false_pos <= 0.001 * sim.is_artefact.size


@pytest.mark.slow
def test_artefact_false_positives_rare_at_low_depth():
    """At 30x most low-VAF artefacts look like errors, but errors shouldn't look like artefacts."""
    est, _ = fitted("wgs_30x", "mixture")
    sim = simulation("wgs_30x")
    flagged = est.sites_report().sort("pos")["artefact_weight"].to_numpy() > 0.5
    assert (flagged & ~sim.is_artefact).sum() <= 0.001 * sim.is_artefact.size


@pytest.mark.parametrize("depth,n", [(30, 50_000), (2000, 5_000)])
def test_mixture_leaves_clean_data_alone(depth, n):
    """With no artefacts the mixture must not eat genuine errors."""
    sim = simulate_sites(n=n, depth=depth, rate=1e-3, artefact_fraction=0.0, seed=1)
    mix = make_estimator(sim.raw_sites)
    raw = make_estimator(sim.raw_sites, artefact_model=False)
    mix_rate = mix.error_table()["error_rate"][-1]
    raw_rate = raw.error_table()["error_rate"][-1]
    assert mix_rate == pytest.approx(raw_rate, rel=0.01)
    assert mix.fit.pi < 1e-3
    assert raw_rate == pytest.approx(sim.truth["overall"], rel=OVERALL_TOL)
