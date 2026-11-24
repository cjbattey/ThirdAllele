"""Parsing of bcftools records, substitution folding, QC flags and edge cases."""

import math

import polars as pl
import pytest

from third_allele import MUTATION_TYPES, QUERY_COLUMNS, ThirdAlleleErrorEstimator


def records(*rows) -> pl.DataFrame:
    """Rows of (pos, ref, alt, gt, ad) on chr1 with QUAL 100."""
    return pl.DataFrame(
        [("chr1", pos, ref, alt, "100", gt, ad) for pos, ref, alt, gt, ad in rows],
        schema=QUERY_COLUMNS,
        orient="row",
    ).with_columns(pl.col("pos").cast(pl.Int64))


def by_pos(df: pl.DataFrame) -> dict:
    return {r["pos"]: r for r in df.iter_rows(named=True)}


class TestParsing:
    def test_germline_and_third_alleles(self, estimator):
        est = estimator(records(
            (1, "C", "T,A", "0/1", "50,48,2"),  # third allele after germline alt
            (2, "C", "A,T", "0/2", "50,1,48"),  # germline alt is the second ALT
            (3, "G", "A", "0|1", "40,40"),  # phased, no third allele
        ))  # fmt: skip
        s = by_pos(est.sites)
        assert (s[1]["germ_alt"], s[1]["n_err"], s[1]["depth"]) == ("T", 2, 100)
        assert (s[2]["germ_alt"], s[2]["n_err"], s[2]["alt_count"]) == ("T", 1, 48)
        assert (s[3]["germ_alt"], s[3]["n_err"]) == ("A", 0)

    @pytest.mark.parametrize(
        "row",
        [
            (1, "C", "T,A", "1/2", "2,50,48"),  # no reference allele
            (1, "C", "T", "1/1", "0,90"),  # hom alt
            (1, "CA", "C", "0/1", "40,40"),  # indel
            (1, "C", "T,A", "0/1", "50,48"),  # AD length mismatch
            (1, "C", "T", "./.", "40,40"),  # missing genotype
        ],
    )
    def test_unusable_records_dropped(self, estimator, row):
        assert estimator(records(row)).sites.height == 0

    def test_non_base_alleles_ignored(self, estimator):
        est = estimator(records((1, "C", "T,<*>", "0/1", "50,48,7")))
        s = by_pos(est.sites)[1]
        assert (s["depth"], s["n_err"]) == (98, 0)


class TestFolding:
    def test_site_alleles_cover_both_third_alleles_with_folded_types(self, estimator):
        est = estimator(records(
            (1, "C", "T,A", "0/1", "50,48,2"),
            (2, "A", "T", "0/1", "40,40"),  # A>T is invisible: only x in {C, G}
            (3, "G", "A", "0/1", "40,40"),  # both purines: everything complemented
        ))  # fmt: skip
        got = {
            (r["pos"], r["x"]): (r["count"], r["type_ref"], r["type_alt"])
            for r in est.site_alleles.iter_rows(named=True)
        }
        assert got == {
            (1, "A"): (2, "C>A", "T>A"),
            (1, "G"): (0, "C>G", "T>G"),
            (2, "C"): (0, "T>G", "T>C"),
            (2, "G"): (0, "T>C", "T>G"),
            (3, "C"): (0, "C>G", "T>G"),
            (3, "T"): (0, "C>A", "T>A"),
        }

    def test_all_types_are_canonical(self, estimator):
        rows = [
            (i + 1, ref, alt, "0/1", "30,30")
            for i, (ref, alt) in enumerate((r, a) for r in "ACGT" for a in "ACGT" if r != a)
        ]
        sa = estimator(records(*rows)).site_alleles
        assert set(sa["type_ref"]) | set(sa["type_alt"]) == set(MUTATION_TYPES)

    def test_error_only_rate_by_hand(self, estimator):
        # 100 C reads, 100 T reads, 4 A reads and nothing else; no artefact model.
        # EM split of the 4 A reads: C>A vs T>A in proportion to their rates.
        # With only one site the rates are not separately identifiable, but the
        # total errors and opportunities are exact.
        est = estimator(records((1, "C", "T,A", "0/1", "100,100,4")), artefact_model=False)
        tab = by_pos_type(est.error_table())
        assert tab["overall"]["errors"] == pytest.approx(4)
        # Each germline read is an opportunity for two classes (x = A, x = G).
        assert tab["overall"]["opportunities"] == pytest.approx(400)
        assert tab["C>G"]["errors"] == 0 and tab["T>G"]["errors"] == 0
        assert tab["C>A"]["errors"] + tab["T>A"]["errors"] == pytest.approx(4)


def by_pos_type(tab: pl.DataFrame) -> dict:
    return {r["mutation_type"]: r for r in tab.iter_rows(named=True)}


class TestQC:
    def test_flags(self, estimator):
        est = estimator(
            records(
                (1, "C", "T", "0/1", "10,9"),  # too shallow
                (2, "C", "T", "0/1", "90,10"),  # VAF 0.1: not het
                (3, "C", "T,A", "0/1", "30,30,3"),  # passes QC, but over the cap
                (4, "C", "T,A", "0/1", "30,30,1"),
                (5, "C", "T", "0/1", "300,300"),  # too deep
            ),
            min_depth=20,
            max_site_depth=500,
            max_errors_per_site=2,
        )
        s = by_pos(est.sites)
        assert [s[p]["pass_qc"] for p in range(1, 6)] == [False, False, True, True, False]
        assert [s[p]["used"] for p in range(1, 6)] == [False, False, False, True, False]
        summary = est.summary()
        assert (summary["pass_qc"], summary["used"], summary["dropped_by_cap"]) == (2, 1, 1)


class TestEdgeCases:
    def test_no_sites(self, estimator):
        empty = pl.DataFrame(schema={c: pl.Utf8 for c in QUERY_COLUMNS}).with_columns(
            pl.col("pos").cast(pl.Int64)
        )
        est = estimator(empty)
        tab = est.error_table()
        assert tab.height == len(MUTATION_TYPES) + 1
        assert all(math.isnan(x) for x in tab["error_rate"])
        assert est.summary()["used"] == 0
        assert est.rate_bias == 0.0

    def test_sites_without_any_errors(self, estimator):
        est = estimator(records(*[(i, "C", "T", "0/1", "40,40") for i in range(1, 50)]))
        tab = by_pos_type(est.error_table())
        assert tab["overall"]["error_rate"] == 0
        assert tab["overall"]["ci_high"] > 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(artefact_tail_quantile=1.0),
            dict(artefact_tail_quantile=0.0),
            dict(artefact_max_vaf=0.0),
            dict(artefact_max_vaf=1.5),
            dict(prior_error_rate=0.0),
            dict(max_em_iterations=0),
        ],
    )
    def test_invalid_parameters(self, tmp_path, kwargs):
        with pytest.raises(ValueError):
            ThirdAlleleErrorEstimator(bam=tmp_path / "a.bam", reference=tmp_path / "r.fa", **kwargs)

    def test_bcftools_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path))
        est = ThirdAlleleErrorEstimator(bam=tmp_path / "a.bam", reference=tmp_path / "r.fa")
        with pytest.raises(RuntimeError, match="bcftools not found"):
            est.raw_sites
