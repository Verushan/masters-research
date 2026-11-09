"""Arm-level summary of convergence_audit.py output (rules: preregistration-2026-11-10-convergence.md).

Per kitchen and arm: how many runs converged / stayed stable / declined / collapsed (with exact
95% intervals), peak and final return, collapse times (Kaplan-Meier median), sudden vs gradual
drift, the test-retest noise floor of the replays, and how often the training curve alone gives
the same class (Cohen's kappa). Arms are compared with a base arm per kitchen: Fisher's exact
test on the collapse rate, a log-rank test on time to collapse, and the exact permutation test
on peak and final return.

    python experiments/analyze_convergence.py experiments/results/convergence/*.json \\
        --base bench_sp-cx-lr --out experiments/results/convergence_summary.json
"""

import argparse
import json
import os.path as osp
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
from compare_arms import permutation_test  # noqa: E402

CLASSES = ("not converged", "stable", "declined", "collapsed")


def clopper_pearson(k, n, a=0.05):
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def kaplan_meier(times, events):
    """times: collapse step or censoring step; events: 1 collapsed, 0 censored."""
    order = np.argsort(times)
    t, e = np.asarray(times, float)[order], np.asarray(events)[order]
    at_risk, s, curve = len(t), 1.0, []
    for ti in np.unique(t):
        d = int(e[t == ti].sum())
        if d:
            s *= 1 - d / at_risk
            curve.append((float(ti), s))
        at_risk -= int((t == ti).sum())
    median = next((ti for ti, si in curve if si <= 0.5), None)
    return curve, median


def logrank(t1, e1, t2, e2):
    t = np.concatenate([t1, t2]).astype(float)
    e = np.concatenate([e1, e2])
    g = np.concatenate([np.zeros(len(t1)), np.ones(len(t2))])
    o_minus_e, var = 0.0, 0.0
    for ti in np.unique(t[e == 1]):
        risk = t >= ti
        n, n1 = risk.sum(), (risk & (g == 0)).sum()
        d = ((t == ti) & (e == 1)).sum()
        d1 = ((t == ti) & (e == 1) & (g == 0)).sum()
        o_minus_e += d1 - d * n1 / n
        if n > 1:
            var += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    if var == 0:
        return None
    return float(stats.chi2.sf(o_minus_e ** 2 / var, 1))


def kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def arm_name(exp):
    return exp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--base", nargs="*", default=["bench_sp-cx-lr"],
                    help="Base arm(s); each arm is compared with the first base of the same stage present")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by_layout = defaultdict(dict)
    noise = defaultdict(list)
    for path in args.files:
        d = json.load(open(path))
        by_layout[d["layout"]][d["exp"]] = d
        for r in d["runs"].values():
            for c in r["checkpoints"]:
                noise[d["layout"]].append(abs(np.mean(c["A"]) - np.mean(c["B"])))

    out = {}
    for layout, arms in sorted(by_layout.items()):
        nf = float(np.percentile(noise[layout], 95))
        print(f"\n== {layout}: replay noise floor (95th pct of |set A - set B| per checkpoint) = {nf:.1f}")
        print(f"   {'arm':32s} {'n':>2s}  " + "  ".join(f"{c:>13s}" for c in CLASSES)
              + "   peak  final  KM-median  sudden  kappa(curve)")
        table = {}
        for exp, d in sorted(arms.items()):
            runs = list(d["runs"].values())
            cls = [r["offline"]["class"] for r in runs]
            cnt = Counter(cls)
            n = len(runs)
            times, events = [], []
            for r in runs:
                o = r["offline"]
                if "collapse_step" in o and o["class"] in ("declined", "collapsed"):
                    times.append(o["collapse_step"])
                    events.append(1)
                else:
                    times.append(r["checkpoints"][-1]["step"])
                    events.append(0)
            km, med = kaplan_meier(times, events)
            curve_cls = [r["from_curve"]["class"] if r.get("from_curve") else "not converged" for r in runs]
            dec = [r for r in runs if r["offline"]["class"] in ("declined", "collapsed")]
            entry = {
                "n": n,
                "counts": {c: cnt.get(c, 0) for c in CLASSES},
                "rate_ci": {c: clopper_pearson(cnt.get(c, 0), n) for c in CLASSES},
                "converged": n - cnt.get("not converged", 0),
                "peak": [r["offline"]["peak"] for r in runs],
                "final": [r["offline"]["final"] for r in runs],
                "collapse_times": times, "collapse_events": events, "km": km, "km_median": med,
                "sudden": sum(r["offline"]["drift"] == "sudden" for r in dec),
                "kappa_curve": kappa(cls, curve_cls),
                "curve_counts": dict(Counter(curve_cls)),
                "per_run": {s: {"offline": r["offline"]["class"], "curve": (r.get("from_curve") or {}).get("class")}
                            for s, r in d["runs"].items()},
            }
            table[exp] = entry
            print(f"   {exp:32s} {n:2d}  " + "  ".join(f"{cnt.get(c, 0):13d}" for c in CLASSES)
                  + f"  {np.mean(entry['peak']):5.0f}  {np.mean(entry['final']):5.0f}"
                  + f"  {med / 1e6 if med else float('nan'):8.1f}M  {entry['sudden']}/{len(dec)}"
                  + f"  {entry['kappa_curve']:.2f} (curve: {dict(Counter(curve_cls))})")
        tests = {}
        stage = lambda e: "s2" if e.startswith("fcp-S2") else "s1"  # noqa: E731
        for exp in table:
            base = next((b for b in args.base if b in table and stage(b) == stage(exp)), None)
            if base is None or base == exp:
                continue
            a, b = table[exp], table[base]
            fail = lambda t: t["counts"]["declined"] + t["counts"]["collapsed"]  # noqa: E731
            _, p_f = stats.fisher_exact([[fail(a), a["n"] - fail(a)], [fail(b), b["n"] - fail(b)]])
            p_lr = logrank(a["collapse_times"], a["collapse_events"], b["collapse_times"], b["collapse_events"])
            dp, pp, *_ = permutation_test(a["peak"], b["peak"])
            df, pf, *_ = permutation_test(a["final"], b["final"])
            tests[f"{exp} vs {base}"] = {"fisher_decline_p": float(p_f), "logrank_p": p_lr, "peak_diff": float(dp),
                                         "peak_p": float(pp), "final_diff": float(df), "final_p": float(pf)}
            print(f"   {exp} vs {base}: declined/collapsed {fail(a)}/{a['n']} vs {fail(b)}/{b['n']} "
                  f"(Fisher p={p_f:.3f}, log-rank p={p_lr if p_lr is None else round(p_lr, 3)}); "
                  f"peak {dp:+.0f} (p={pp:.3f}), final {df:+.0f} (p={pf:.3f})")
        out[layout] = {"noise_floor": nf, "arms": table, "tests": tests}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
