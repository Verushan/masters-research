"""Trained without swaps, tested with them: the 2x2 the supervisors asked for (15 Oct).

Reads swap cross-play records (pipelines/fillin-crossplay.slurm with SWAP_STEP):
every agent plays every held-out partner in both seats, and at the swap step
the partner is replaced by a different held-out partner without warning.

Per training run (the unit), averaged over partners, seats and episodes:
    before   sparse return over the steps before the swap
    after    sparse return over the steps from the swap on
    kept     after / before: how much of its pre-swap performance it keeps

Arms are grouped by experiment name (the run suffix _s<seed> removed). The
pre-registered comparisons (experiments/report/preregistration-2026-10-15-swaps.md):
    H1   C2'-ns vs A-ns on `after`: among agents trained without swaps, does
         MORL hold up better when the partner changes?
    H2   the swap-training benefit, (A - A-ns) vs (C2' - C2'-ns) on `after`:
         does MORL need swap training less than the hand-shaped reward does?

    python experiments/analyze_swaps.py \\
        --records experiments/results/cross_play_unident_s_hsp_noswap_swap200.json \\
        --out experiments/results/swaps_unident_s.json
"""

import argparse
import json
import os.path as osp
import re
import sys
from collections import defaultdict

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
from compare_arms import permutation_test  # noqa: E402

A, ANS, C, CNS = "s2_scripted-hand", "s2_scripted-handns", "s2_scripted-noann2", "s2_scripted-noann2ns"
LABELS = {
    A: "A: hand-shaped, trained with swaps",
    ANS: "A-ns: hand-shaped, no swaps",
    C: "C2': MORL, trained with swaps",
    CNS: "C2'-ns: MORL, no swaps",
    "s2_bench_sp": "Usual stage-2",
    "bench_sp": "Stage-1 self-play",
}


def per_run(records_paths):
    runs = defaultdict(lambda: defaultdict(list))
    swap_step = None
    for path in records_paths:
        d = json.load(open(path))
        swap_step = d.get("swap_step") or swap_step
        for r in d["records"]:
            if "partner_seat" not in r:
                continue
            ego = r["agent1"] if r["partner_seat"] == 0 else r["agent0"]
            after = r["eval_ep_sparse_r_after_swap"]
            runs[ego]["before"].append(r["eval_ep_sparse_r"] - after)
            runs[ego]["after"].append(after)
    out = {}
    for ego, v in runs.items():
        before, after = float(np.mean(v["before"])), float(np.mean(v["after"]))
        out[ego] = {"before": before, "after": after, "kept": after / before if before > 0 else float("nan"),
                    "games": len(v["after"])}
    return out, swap_step


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs, swap_step = per_run(args.records)

    arms = defaultdict(dict)
    for ego, v in runs.items():
        arms[re.sub(r"_s\d+$", "", ego)][ego] = v
    table = {}
    for arm, members in arms.items():
        col = lambda k: [m[k] for m in members.values()]  # noqa: E731
        table[arm] = {"label": LABELS.get(arm, arm), "n": len(members),
                      **{k: float(np.nanmean(col(k))) for k in ("before", "after", "kept")},
                      "after_sd": float(np.std(col("after"), ddof=1)) if len(members) > 1 else None,
                      "runs": members}

    tests = {}
    after = {arm: [m["after"] for m in t["runs"].values()] for arm, t in table.items()}
    if CNS in after and ANS in after:
        diff, p, *_ = permutation_test(after[CNS], after[ANS])
        tests["H1: C2'-ns vs A-ns (after swap)"] = {"diff": float(diff), "p": float(p)}
    if all(k in after for k in (A, ANS, C, CNS)):
        # Difference of differences; permutation within each reward, run as the unit.
        rng = np.random.default_rng(0)

        def contrast(a, ans, c, cns):
            return (np.mean(a) - np.mean(ans)) - (np.mean(c) - np.mean(cns))

        obs = contrast(after[A], after[ANS], after[C], after[CNS])
        hand, morl = after[A] + after[ANS], after[C] + after[CNS]
        n_a, n_c = len(after[A]), len(after[C])
        null = []
        for _ in range(20000):
            h, m = rng.permutation(hand), rng.permutation(morl)
            null.append(contrast(h[:n_a], h[n_a:], m[:n_c], m[n_c:]))
        p = float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
        tests["H2: swap-training benefit, hand minus MORL (after swap)"] = {"diff": float(obs), "p": p}
    for arm in (ANS, CNS):
        base = arm[: -len("ns")]
        if base in after and arm in after:
            diff, p, *_ = permutation_test(after[base], after[arm])
            tests[f"swap training helps ({LABELS[base].split(':')[0]})"] = {"diff": float(diff), "p": float(p)}

    out = {"swap_step": swap_step, "arms": table, "tests": tests}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"swap at step {swap_step}; per run, averaged over partners, seats and episodes")
    for arm, t in sorted(table.items(), key=lambda kv: -kv[1]["after"]):
        print(f"  {t['label']:40s} n={t['n']:2d}  before {t['before']:6.1f}  after {t['after']:6.1f}  kept {t['kept']:.2f}")
    for name, t in tests.items():
        print(f"  {name:55s} diff {t['diff']:+6.1f}  p={t['p']:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
