"""Oracle-bound ensemble report — best-per-seed across all completed runs.

Reads eval logs from prior runs, picks the best return per seed across all,
reports the upper-bound mean an ensemble could achieve with perfect routing.
This is what 'pick the right specialist for each config' would max at.
"""

from __future__ import annotations

# Per-seed-best across all completed runs, hand-collated from eval logs.
# Each entry: (run_name, return, max_coverage)
PER_SEED_BEST = {
    0: ("W",  80.430, 0.516),
    1: ("AB", 48.596, 0.498),
    2: ("T2", 39.142, 0.270),
    3: ("T1", 23.335, 0.216),
    4: ("T1", 16.857, 0.114),
    5: ("Z",  12.064, 0.147),
    6: ("U",  73.820, 0.569),
    7: ("T2",  0.336, 0.054),
    8: ("T2", 32.012, 0.206),
    9: (None,  0.000, 0.000),  # universally failing
}


def main():
    print("=== Oracle ensemble (best-per-seed) report ===\n")
    print(f"{'Seed':<6}{'Run':<6}{'Return':>10}{'MaxCov':>10}")
    total_return = 0.0
    total_cov = 0.0
    n = len(PER_SEED_BEST)
    for seed, (run, ret, cov) in PER_SEED_BEST.items():
        run_str = run if run else "—"
        print(f"{seed:<6}{run_str:<6}{ret:>10.3f}{cov:>10.3f}")
        total_return += ret
        total_cov += cov
    print()
    print(f"  Oracle mean return     : {total_return / n:+.3f}")
    print(f"  Oracle mean max_cov    : {total_cov / n:.3f}")
    print()
    print("Compare to single best:")
    print(f"  T2 (best single mean)  : +16.450  (cov ~0.135)")
    print(f"  Oracle upper bound     : {total_return / n:+.3f}  (cov {total_cov / n:.3f})")
    print(f"  Lift                   : {(total_return / n) / 16.450:.2f}x")


if __name__ == "__main__":
    main()
