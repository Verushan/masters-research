"""Does knowing who your partner is help at all?

Reads the per-rung, per-seed records `pid-ceiling.slurm` writes and compares
the three ladder rungs against the population they trained on, with the true
partner ids:

    rung1  the actor never sees the id (control)
    rung2  the actor sees the raw scalar id
    rung3  the actor sees a one-hot over the population

The rungs are identical in population, reward, budget and schedule, so the gap
between them is the value of the partner's identity and nothing else.

This is an ORACLE CEILING and not a zero-shot result. A held-out partner has no
id the agent was ever trained on, so these numbers cannot be reported as ZSC
performance. What they bound is how much headroom any partner-inference method
could possibly recover: if being handed the true identity is worth nothing,
then inferring it is worth nothing too, and an adaptation claim should become a
robustness claim instead.

The unit of replication is the training run, as in compare_arms.py, and with
three or four runs per rung the exact permutation test is the honest one --
including its floor, since at these group sizes p<0.05 can be unreachable
whatever the data say.

    python experiments/analyze_pid_ceiling.py --results experiments/results
"""

import argparse
import glob
import gzip
import itertools
import json
import os
import re
from collections import defaultdict

import numpy as np

RUNGS = ["rung1", "rung2", "rung3"]
LABEL = {
    "rung1": "control: actor never sees the id",
    "rung2": "actor sees the raw scalar id",
    "rung3": "actor sees a one-hot over the population",
}
PATTERN = re.compile(
    r"pid_ceiling_(?P<layout>.+?)_(?P<arm>.+?)_(?P<rung>rung\d)_s(?P<seed>[0-9a-z]+)\.json"
)


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        payload = json.load(handle)
    return payload["records"] if isinstance(payload, dict) else payload


def permutation_test(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    observed = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    n = len(a)
    diffs = []
    for idx in itertools.combinations(range(len(pool)), n):
        mask = np.zeros(len(pool), bool)
        mask[list(idx)] = True
        diffs.append(pool[mask].mean() - pool[~mask].mean())
    diffs = np.array(diffs)
    p = float(np.mean(np.abs(diffs) >= abs(observed) - 1e-12))
    # A split and its mirror are both enumerated only when the groups are
    # the same size; with unequal sizes each split appears once.
    floor = (2.0 if len(a) == len(b) else 1.0) / len(diffs)
    return observed, p, floor, len(diffs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="experiments/results")
    parser.add_argument("--layout", default="random0")
    parser.add_argument("--arm", default="bench_sp")
    args = parser.parse_args()

    by_rung = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(args.results, "pid_ceiling_*.json*"))):
        m = PATTERN.match(os.path.basename(path).replace(".gz", ""))
        if not m or m["layout"] != args.layout or m["arm"] != args.arm:
            continue
        rows = load(path)
        returns = [r.get("eval_ep_sparse_r", 0.0) for r in rows]
        by_rung[m["rung"]][m["seed"]] = float(np.mean(returns)) if returns else float("nan")

    if not by_rung:
        raise SystemExit(f"no pid_ceiling records for {args.layout}/{args.arm}")

    print("=" * 78)
    print(f"partner-identity oracle ceiling -- {args.layout} / {args.arm}")
    print("scored against the training population, with true ids")
    print("=" * 78)
    print(f"\n{'rung':7s} {'n':>2s} {'mean':>7s} {'sd':>6s}  per-seed")
    print("-" * 78)
    for rung in RUNGS:
        seeds = by_rung.get(rung, {})
        if not seeds:
            print(f"{rung:7s}  0        -      -  (no records)")
            continue
        v = np.array(list(seeds.values()))
        sd = v.std(ddof=1) if len(v) > 1 else float("nan")
        detail = ", ".join(f"{k}={seeds[k]:.1f}" for k in sorted(seeds))
        print(f"{rung:7s} {len(v):2d} {v.mean():7.1f} {sd:6.1f}  {detail}")
    for rung in RUNGS:
        if by_rung.get(rung):
            print(f"  {rung}: {LABEL[rung]}")

    base = by_rung.get("rung1")
    if not base or len(base) < 2:
        print("\nno usable rung1 control; the ceiling cannot be read without it")
        return

    print("\nagainst rung1 (exact permutation over training runs)")
    print("-" * 78)
    for rung in ("rung2", "rung3"):
        seeds = by_rung.get(rung, {})
        if len(seeds) < 2:
            print(f"{rung:7s} n={len(seeds)}, too few runs to test")
            continue
        diff, p, floor, n_perm = permutation_test(
            list(seeds.values()), list(base.values())
        )
        print(
            f"{rung:7s} diff {diff:+7.1f}  p={p:.3f} "
            f"({n_perm} perms, min possible p={floor:.2f})"
        )
        if floor > 0.05:
            print(
                f"{'':7s}   ^ these group sizes cannot reach p<0.05 whatever the data"
            )

    best = max(
        (r for r in RUNGS if by_rung.get(r)),
        key=lambda r: np.mean(list(by_rung[r].values())),
    )
    gap = np.mean(list(by_rung[best].values())) - np.mean(list(base.values()))
    print(
        f"\nbest rung is {best}, {gap:+.1f} against the control.\n"
        "That gap is the entire headroom a partner-inference method could recover:\n"
        "an agent that perfectly identified its partner could do no better than an\n"
        "agent simply told who it was."
    )
    print()


if __name__ == "__main__":
    main()
