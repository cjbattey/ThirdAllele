"""Unit tests for the statistical helpers."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import binom, poisson

from third_allele.estimator import (
    artefact_floor_reads,
    artefact_posterior,
    clopper_pearson,
    limit_of_detection,
    log_artefact_likelihood,
    log_poisson,
    truncation_loss,
)


class TestClopperPearson:
    def test_known_value(self):
        # Reference values from R: binom.test(5, 100)$conf.int
        lo, hi = clopper_pearson(5, 100)
        assert lo == pytest.approx(0.01643188, rel=1e-5)
        assert hi == pytest.approx(0.11283491, rel=1e-5)

    def test_edges(self):
        assert clopper_pearson(0, 50)[0] == 0.0
        assert clopper_pearson(50, 50)[1] == 1.0
        assert all(math.isnan(x) for x in clopper_pearson(0, 0))

    def test_fractional_counts_interpolate(self):
        lo4, hi4 = clopper_pearson(4, 100)
        lo, hi = clopper_pearson(4.5, 100)
        lo5, hi5 = clopper_pearson(5, 100)
        assert lo4 < lo < lo5 and hi4 < hi < hi5


class TestLimitOfDetection:
    @pytest.mark.parametrize("rate,depth", [(1e-3, 200), (1e-3, 2000), (5e-7, 2000), (1e-2, 30)])
    def test_definition(self, rate, depth):
        alpha, power = 0.05, 0.95
        k, lod = limit_of_detection(rate, depth, alpha, power)
        # k is the smallest count whose false-positive probability is <= alpha
        assert binom.sf(k - 1, depth, rate) <= alpha
        assert binom.sf(k - 2, depth, rate) > alpha or k == 1
        # at the LoD VAF, k alt reads are reached with probability `power`
        assert binom.sf(k - 1, depth, lod) == pytest.approx(power, abs=1e-6)

    def test_lod_rises_with_error_rate(self):
        lods = [limit_of_detection(r, 1000)[1] for r in (1e-5, 1e-4, 1e-3, 1e-2)]
        assert lods == sorted(lods)

    def test_degenerate(self):
        assert limit_of_detection(1e-3, 0) == (0, pytest.approx(math.nan, nan_ok=True))
        assert math.isnan(limit_of_detection(math.nan, 100)[1])
        k, lod = limit_of_detection(0.9, 5)  # threshold above depth
        assert k > 5 and math.isnan(lod)


class TestTruncationLoss:
    @pytest.mark.parametrize("lam,k", [(0.2, 1), (1.0, 1), (1.0, 3), (3.0, 5)])
    def test_inverts_truncated_mean(self, lam, k):
        # Mean of a Poisson(lam) conditioned on <= k, then recover the lost share.
        c = np.arange(k + 1)
        p = poisson.pmf(c, lam)
        mean_kept = float((c * p).sum() / p.sum())
        expected = 1 - poisson.cdf(k - 1, lam)
        assert truncation_loss(mean_kept, k) == pytest.approx(expected, rel=1e-6)

    def test_no_loss_cases(self):
        assert truncation_loss(0.0, 2) == 0.0
        assert truncation_loss(0.5, 0) == 0.0

    def test_impossible_mean_is_nan(self):
        assert math.isnan(truncation_loss(1.5, 1))


class TestArtefactLikelihood:
    @pytest.mark.parametrize(
        "c,n,lo,hi",
        [
            (0, 30, 0.1, 0.5),
            (3, 30, 0.1, 0.5),
            (12, 30, 0.1, 0.5),
            (3, 2000, 0.0015, 0.5),  # mode below the interval: survival branch
            (200, 2000, 0.0015, 0.5),
        ],
    )
    def test_matches_numerical_integral(self, c, n, lo, hi):
        integral, _ = quad(lambda f: binom.pmf(c, n, f), lo, hi, points=[c / n], limit=200)
        expected = math.log(integral / (hi - lo))
        got = float(log_artefact_likelihood(c, n, lo, hi))
        assert got == pytest.approx(expected, rel=1e-6, abs=1e-6)

    def test_deep_tail_closed_form(self):
        # c = 1: integral of n f (1-f)^(n-1) df = [(1-f)^n - n/(n+1) (1-f)^(n+1)] evaluated
        # from hi to lo. The mass is ~1e-307 here, where numerical quadrature fails.
        n, lo, hi = 2000, 0.3, 0.5

        def log_antiderivative(f):
            return n * math.log(1 - f) + math.log(1 - n / (n + 1) * (1 - f))

        log_mass = log_antiderivative(lo) + math.log1p(-math.exp(log_antiderivative(hi) - log_antiderivative(lo)))
        expected = log_mass - math.log(hi - lo)
        assert float(log_artefact_likelihood(1, n, lo, hi)) == pytest.approx(expected, rel=1e-9)

    def test_empty_interval_is_impossible(self):
        assert log_artefact_likelihood(3, 30, 0.5, 0.5) == -np.inf
        assert log_artefact_likelihood(3, 30, 0.6, 0.5) == -np.inf

    def test_broadcasts(self):
        out = log_artefact_likelihood(np.array([0, 5, 10]), 100, 0.03, 0.5)
        assert out.shape == (3,) and np.all(np.isfinite(out))


class TestMixtureHelpers:
    def test_log_poisson(self):
        c, lam = np.array([0, 1, 5]), np.array([0.1, 2.0, 3.0])
        np.testing.assert_allclose(log_poisson(c, lam), poisson.logpmf(c, lam))

    def test_posterior(self):
        assert np.all(artefact_posterior(np.zeros(3), np.zeros(3), 0.0) == 0)
        np.testing.assert_allclose(artefact_posterior(np.log([0.2]), np.log([0.6]), 0.5), [0.75])

    def test_floor(self):
        np.testing.assert_array_equal(artefact_floor_reads(np.array([1e-6, 0.1]), 3, 0.999), [3, 3])
        # high lambda: floor is the Poisson quantile, well above min_reads
        assert artefact_floor_reads(np.array([10.0]), 3, 0.999)[0] == poisson.ppf(0.999, 10)
