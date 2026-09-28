"""Zero-shot score per arm from an analyze_crossplay metrics file, grouped by agent name.

analyze_crossplay.py only groups the arm names it knows, so newer stage-2 agents
(s2_scripted-*) come out ungrouped. This groups every non-partner agent by its
name minus the `_s{seed}` suffix, prints mean / SD / worst / best zero-shot
return per arm, and compares each arm with --base by the exact permutation test
on seed-level values (the training run is the unit of replication).

    python experiments/zsc_by_arm.py experiments/results/metrics_unident_s_hsp_scripted.json \\
        --base s2_scripted-hand
"""

import argparse
import json
import os.path as osp
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from compare_arms import permutation_test  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("metrics")
    ap.add_argument("--base", default=None, help="Arm every other is compared against")
    args = ap.parse_args()

    d = json.load(open(args.metrics))
    partners = set(d["partners"])
    arms = defaultdict(list)
    for agent, v in d["per_agent"].items():
        if agent in partners or v.get("zsc_mean") is None:
            continue
        arms[re.sub(r"_s\d+(?:r\d+)?$", "", agent)].append(v["zsc_mean"])

    print(f"{len(partners)} held-out partners ({d['partner_group']})")
    width = max(len(a) for a in arms) + 2
    for arm, vals in sorted(arms.items(), key=lambda kv: -np.mean(kv[1])):
        x = np.array(vals)
        sd = x.std(ddof=1) if len(x) > 1 else float("nan")
        print(f"{arm:{width}s} n={len(x):2d}  mean {x.mean():6.1f}  sd {sd:5.1f}  worst {x.min():6.1f}  best {x.max():6.1f}")
    if args.base in arms:
        print(f"\nagainst {args.base} (exact permutation test, seed level)")
        for arm, vals in arms.items():
            if arm == args.base or len(vals) < 2:
                continue
            diff, p, *_ = permutation_test(vals, arms[args.base])
            print(f"  {arm:{width}s} {diff:+7.1f}  p={p:.3f}")


if __name__ == "__main__":
    main()
