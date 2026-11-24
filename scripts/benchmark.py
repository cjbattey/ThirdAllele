"""Compare estimators on simulated het sites with known error rates and artefacts.

Prints estimated/true rate per class for each regime (1.000 is perfect).
Pass assertions live in tests/test_simulation.py; this is for eyeballing.

    python scripts/benchmark.py [--seed N]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from third_allele import MUTATION_TYPES, ThirdAlleleErrorEstimator
from third_allele.simulate import simulate_sites

REGIMES = {
    "WGS 30x, r=1e-3": dict(n=200_000, depth=30, rate=1e-3, artefact_fraction=0.01),
    "raw capture 2000x, r=1e-3": dict(n=20_000, depth=2000, rate=1e-3, artefact_fraction=0.02),
    "duplex 2000x, r=5e-7": dict(n=200_000, depth=2000, rate=5e-7, artefact_fraction=0.005),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    for name, p in REGIMES.items():
        sim = simulate_sites(**p, seed=args.seed)
        print(f"\n=== {name}  (sites={p['n']}, artefact sites={sim.is_artefact.sum()})")
        configs = {
            "mixture": {},
            "mixture+prior": dict(prior_error_rate=p["rate"]),
            "legacy k=1": dict(artefact_model=False, max_errors_per_site=1),
            "no filter": dict(artefact_model=False),
        }
        out = {}
        for label, kw in configs.items():
            est = ThirdAlleleErrorEstimator(bam=Path("-"), reference=Path("-"), **kw)
            est.raw_sites = sim.raw_sites
            t = time.time()
            tab = est.error_table()
            dt = time.time() - t
            out[label] = dict(zip(tab["mutation_type"], tab["error_rate"]))
            extra = ""
            if label.startswith("mixture"):
                s = est.summary()
                extra = (
                    f" pi={s['artefact_fraction']:.2e} art_sites={s['artefact_sites']}"
                    f" art_reads={s['artefact_reads']:.0f}/{s['third_allele_reads']}"
                    f" bias={est.rate_bias:+.2%}"
                )
            print(f"  {label:14s} {dt:5.1f}s{extra}")
        print(f"  {'type':8s} {'truth':>9s} " + " ".join(f"{k:>14s}" for k in out))
        for t in MUTATION_TYPES + ["overall"]:
            tr = sim.truth[t]
            print(f"  {t:8s} {tr:9.2e} " + " ".join(f"{out[k][t] / tr:14.3f}" for k in out))


if __name__ == "__main__":
    main()
