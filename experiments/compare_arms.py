"""Compare stage-2 arms with the training run as the unit of replication.

Why this exists, stated plainly because it is the difference between a claim
that survives examination and one that does not.

The earlier comparison averaged each arm's seeds together and then ran a
Wilcoxon test paired over the sixteen held-out partners, reporting p = 0.034
for bench_morl over bench_sp on unident_s. That test is answering a narrower
question than the one it was being read as. Its unit of replication is the
partner, and the partners are a *fixed evaluation set* -- the same sixteen
agents appear in every arm's evaluation. What varies when you ask "does a MORL
population train a better stage-2 agent" is the stage-2 training run. Pairing
over partners while holding three seeds treats the seed-to-seed variance as
zero, and on unident_s that variance is enormous: s2_bench_morl_ad's three
seeds score 174.4, 116.9 and 33.1.

So this module reports two different things and never conflates them:

within-agent   the partner-paired test. Legitimate, and worth reporting, but it
               generalises to *these particular trained agents*, not to the
               method that produced them. Phrase it as "this agent beat that
               agent across the partner set".
between-arm    the seed-level test. This is the one that licenses a claim about
               the training method. With three to six seeds per arm it is
               nearly always underpowered, so it is reported as an effect size
               with a confidence interval and an explicit power statement
               rather than as a bare p-value.

An exact permutation test is used between arms rather than a t-test: with n of
3 to 6 the normal approximation is doing real work that the data cannot
support, while a permutation over all C(n1+n2, n1) label assignments is exact
under the null of exchangeability and costs nothing at this size.

    python experiments/compare_arms.py \
        --metrics experiments/results/metrics_unident_s_s2_hsp.json --label unident_s
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ARMS = ["s2_bench_sp", "s2_bench_morl", "s2_bench_morl_ad", "s2_mixed"]
# Power for a two-sided two-sample test at alpha=.05, 80% power:
#   n per group ~= 2 * (z_{.975} + z_{.80})^2 * sd^2 / delta^2
POWER_CONST = 2 * (1.959964 + 0.841621) ** 2


def seed_values(metrics, arm):
    """Per-seed ZSC mean for one arm. One entry per stage-2 training run."""
    pa = metrics["per_agent"]
    return {a: pa[a]["zsc_mean"] for a, d in pa.items() if d["group"] == arm}


def permutation_test(a, b, n_max=200000, seed=0):
    """Exact two-sided permutation test on the difference of means.

    Enumerates every split when the number of them is manageable, which for the
    group sizes here it always is; falls back to sampling only if not.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    observed = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    n = len(a)
    total = 1
    for i in range(n):
        total = total * (len(pool) - i) // (i + 1)
    if total <= n_max:
        diffs = []
        for idx in itertools.combinations(range(len(pool)), n):
            mask = np.zeros(len(pool), dtype=bool)
            mask[list(idx)] = True
            diffs.append(pool[mask].mean() - pool[~mask].mean())
        diffs = np.array(diffs)
        exact = True
    else:
        rng = np.random.default_rng(seed)
        diffs = np.array(
            [
                (lambda p: p[:n].mean() - p[n:].mean())(rng.permutation(pool))
                for _ in range(n_max)
            ]
        )
        exact = False
    p = float(np.mean(np.abs(diffs) >= abs(observed) - 1e-12))
    # The floor a permutation test can reach at these group sizes. With three
    # seeds against three there are twenty splits, and the most extreme
    # separation the data could possibly show still leaves two of them at least
    # as extreme (the split and its mirror), so p can never fall below 0.1 --
    # significance at .05 is unreachable in principle, not merely unattained.
    p_floor = float(2.0 / len(diffs)) if exact else 1.0 / n_max
    return observed, p, exact, len(diffs), p_floor


def bootstrap_ci(a, b, n_boot=20000, seed=0):
    """Percentile CI for the difference of arm means."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    draws = [
        rng.choice(a, len(a), replace=True).mean()
        - rng.choice(b, len(b), replace=True).mean()
        for _ in range(n_boot)
    ]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def partner_paired(metrics, arm_a, arm_b):
    """The within-agent test: arm means across the shared partner set."""
    pa = metrics["per_agent"]
    ms_a = [a for a, d in pa.items() if d["group"] == arm_a]
    ms_b = [a for a, d in pa.items() if d["group"] == arm_b]
    if not ms_a or not ms_b:
        return None
    partners = sorted(set(pa[ms_a[0]]["vs_partners"]) & set(pa[ms_b[0]]["vs_partners"]))
    va = np.array([[pa[m]["vs_partners"][p] for p in partners] for m in ms_a]).mean(0)
    vb = np.array([[pa[m]["vs_partners"][p] for p in partners] for m in ms_b]).mean(0)
    try:
        _, p = wilcoxon(va, vb)
    except ValueError:
        p = float("nan")
    return va.mean() - vb.mean(), int((va > vb).sum()), len(partners), p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Agent names to drop, e.g. a run established as failed. Excluding a "
        "run changes the result, so anything dropped here belongs in the report "
        "with its reason.",
    )
    args = parser.parse_args()

    metrics = json.load(open(args.metrics))
    label = args.label or Path(args.metrics).stem
    excluded = set(args.exclude)

    print("=" * 78)
    print(f"stage-2 arm comparison -- {label}")
    print(f"unit of replication: the stage-2 training run (seed)")
    if excluded:
        print(f"excluded: {', '.join(sorted(excluded))}")
    print("=" * 78)

    values = {}
    for arm in ARMS:
        vals = {k: v for k, v in seed_values(metrics, arm).items() if k not in excluded}
        if vals:
            values[arm] = vals

    print(f"\n{'arm':22s} {'n':>2s} {'mean':>7s} {'sd':>7s}   per-seed")
    print("-" * 78)
    for arm, vals in values.items():
        v = np.array(list(vals.values()))
        sd = v.std(ddof=1) if len(v) > 1 else float("nan")
        print(
            f"{arm:22s} {len(v):2d} {v.mean():7.1f} {sd:7.1f}   "
            f"{', '.join(f'{x:.1f}' for x in sorted(v, reverse=True))}"
        )

    sds = [np.array(list(v.values())).std(ddof=1) for v in values.values() if len(v) > 1]
    pooled = float(np.sqrt(np.mean(np.square(sds)))) if sds else float("nan")
    print(f"\npooled within-arm sd: {pooled:.1f}")
    if not np.isnan(pooled):
        for delta in (20, 30, 40):
            print(
                f"  n needed per arm to detect a {delta}-point difference "
                f"at 80% power: {np.ceil(POWER_CONST * pooled**2 / delta**2):.0f}"
            )

    base = "s2_bench_sp"
    if base not in values:
        return
    print(f"\nbetween-arm, against {base} (exact permutation on seed means)")
    print("-" * 78)
    for arm, vals in values.items():
        if arm == base:
            continue
        a = list(vals.values())
        b = list(values[base].values())
        if len(a) < 2 or len(b) < 2:
            print(f"{arm:22s} n too small for a test (n={len(a)} vs {len(b)})")
            continue
        diff, p, exact, n_perm, p_floor = permutation_test(a, b)
        lo, hi = bootstrap_ci(a, b)
        kind = "exact" if exact else "sampled"
        floor = f", min possible p={p_floor:.2f}" if exact else ""
        print(
            f"{arm:22s} diff {diff:+7.1f}  95% CI [{lo:+.1f}, {hi:+.1f}]  "
            f"p={p:.3f} ({kind}, {n_perm} perms{floor})"
        )
        if exact and p_floor > 0.05:
            print(
                f"{'':22s}   ^ at n={len(a)} vs {len(b)} no result can reach "
                f"p<0.05; the test cannot support a claim either way"
            )
        if min(len(a), len(b)) < 5:
            print(
                f"{'':22s}   ^ the bootstrap CI resamples {min(len(a), len(b))} "
                f"values and is anticonservative here; prefer the permutation p"
            )

    print(f"\nwithin-agent, against {base} (partner-paired; describes these agents,")
    print("not the method that produced them)")
    print("-" * 78)
    for arm in values:
        if arm == base:
            continue
        res = partner_paired(metrics, arm, base)
        if res:
            d, better, n_p, p = res
            print(
                f"{arm:22s} diff {d:+7.1f}  better on {better}/{n_p} partners  "
                f"Wilcoxon p={p:.3f}"
            )
    print()


if __name__ == "__main__":
    main()
