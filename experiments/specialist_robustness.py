"""Is "train against specialists" a robust result, or an average hiding a few partners?

The headline is arm A's zero-shot mean against the usual stage-2 agent's. This
takes that number apart three ways, using the cross-play matrices already on disk:

1. Per held-out partner. For each of the 16 HSP partners, A's mean over its
   runs against the usual agent's, with a seed-level permutation test. Also the
   number of partners where A is ahead. A result carried by a handful of
   partners would show here.
2. Stochastic actions. The canonical pass is deterministic (argmax). The
   stochastic pass samples actions, which is how the agents were trained; the
   ranking should not depend on the pass.
3. Worst case. Each run's worst partner, and how many of A's runs beat the usual
   agent's best run.

    python experiments/specialist_robustness.py \\
        --metrics experiments/results/metrics_unident_s_hsp_scripted2.json \\
        --arm s2_scripted-hand --ref s2_bench_sp --out experiments/results/specialist_robustness_unident_s.json
"""

import argparse
import json
import os.path as osp
import re
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from compare_arms import permutation_test  # noqa: E402


def runs(d, arm):
    rx = re.compile(rf"^{re.escape(arm)}_s\d+(?:r\d+)?$")
    return sorted(a for a in d["per_agent"] if rx.match(a))


def stochastic_zsc(d, agent, partners):
    """Mean over partners of the stochastic pass, both seat orders."""
    pm = d.get("pair_matrix_stochastic") or {}
    vals = []
    for p in partners:
        cell = [pm[k]["mean"] for k in (f"{agent}|{p}", f"{p}|{agent}") if k in pm]
        if cell:
            vals.append(np.mean(cell))
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--arm", default="s2_scripted-hand")
    ap.add_argument("--ref", default="s2_bench_sp")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.load(open(args.metrics))
    partners = sorted(d["partners"], key=lambda p: int(re.search(r"hsp(\d+)", p).group(1)))
    A, R = runs(d, args.arm), runs(d, args.ref)
    print(f"{args.arm}: {len(A)} runs, {args.ref}: {len(R)} runs, {len(partners)} partners")

    per_partner = []
    for p in partners:
        a = [d["per_agent"][x]["vs_partners"][p] for x in A]
        r = [d["per_agent"][x]["vs_partners"][p] for x in R]
        diff, pval, *_ = permutation_test(a, r)
        per_partner.append(
            {"partner": p, "arm_mean": float(np.mean(a)), "ref_mean": float(np.mean(r)), "diff": float(diff), "p": float(pval)}
        )
    ahead = sum(x["diff"] > 0 for x in per_partner)
    sig_ahead = sum(x["diff"] > 0 and x["p"] < 0.05 for x in per_partner)
    sig_behind = sum(x["diff"] < 0 and x["p"] < 0.05 for x in per_partner)

    det_a = [d["per_agent"][x]["zsc_mean"] for x in A]
    det_r = [d["per_agent"][x]["zsc_mean"] for x in R]
    sto_a = [stochastic_zsc(d, x, partners) for x in A]
    sto_r = [stochastic_zsc(d, x, partners) for x in R]
    det = permutation_test(det_a, det_r)
    sto = permutation_test(sto_a, sto_r) if None not in sto_a + sto_r else (None, None)
    worst_a = [d["per_agent"][x]["zsc_worst"] for x in A]
    worst_r = [d["per_agent"][x]["zsc_worst"] for x in R]
    beats_best = sum(v > max(det_r) for v in det_a)

    out = {
        "metrics": osp.basename(args.metrics),
        "arm": args.arm,
        "ref": args.ref,
        "n_arm": len(A),
        "n_ref": len(R),
        "per_partner": per_partner,
        "partners_ahead": ahead,
        "partners_sig_ahead": sig_ahead,
        "partners_sig_behind": sig_behind,
        "deterministic": {"arm": float(np.mean(det_a)), "ref": float(np.mean(det_r)), "diff": float(det[0]), "p": float(det[1])},
        "stochastic": {
            "arm": float(np.mean(sto_a)) if sto[0] is not None else None,
            "ref": float(np.mean(sto_r)) if sto[0] is not None else None,
            "diff": float(sto[0]) if sto[0] is not None else None,
            "p": float(sto[1]) if sto[0] is not None else None,
        },
        "worst_partner": {"arm": float(np.mean(worst_a)), "ref": float(np.mean(worst_r))},
        "arm_runs_beating_ref_best_run": beats_best,
        "runs": {"arm": dict(zip(A, det_a)), "ref": dict(zip(R, det_r))},
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\nper partner ({args.arm} - {args.ref}, seed-level permutation test)")
    for x in per_partner:
        mark = "*" if x["p"] < 0.05 else " "
        print(f"  {x['partner']:22s} {x['arm_mean']:6.1f} vs {x['ref_mean']:6.1f}  {x['diff']:+7.1f}  p={x['p']:.3f}{mark}")
    print(f"ahead on {ahead}/{len(partners)} partners ({sig_ahead} significantly); significantly behind on {sig_behind}")
    print(f"deterministic: {np.mean(det_a):.1f} vs {np.mean(det_r):.1f}  diff {det[0]:+.1f} p={det[1]:.4f}")
    if sto[0] is not None:
        print(f"stochastic:    {np.mean(sto_a):.1f} vs {np.mean(sto_r):.1f}  diff {sto[0]:+.1f} p={sto[1]:.4f}")
    print(f"worst partner per run, mean: {np.mean(worst_a):.1f} vs {np.mean(worst_r):.1f}")
    print(f"{beats_best}/{len(A)} {args.arm} runs beat the best {args.ref} run ({max(det_r):.1f})")


if __name__ == "__main__":
    main()
