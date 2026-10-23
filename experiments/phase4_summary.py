"""Phase 4 on the complex-dish kitchens: zero-shot and BR-Prox per stage-2 arm.

Reads the cross-play metrics (fillin-crossplay.slurm, HELDOUT_ONLY, sampled
actions, both seats) and the best-response values. Per training run (the unit),
zero-shot is the mean return with each held-out partner, symmetrised over the
two seats; BR-Prox divides that by max(best response, best pool agent) per
partner, over partners that have a best response. Arms are compared with the
exact permutation test on run-level values.

    python experiments/phase4_summary.py --budget 2m     # -> results/phase4_summary_2m.json
    python experiments/phase4_summary.py --budget 5m
"""

import argparse
import json
import os.path as osp
import re
import sys
from collections import defaultdict

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
RESULTS = osp.join(HERE, "results")
sys.path.insert(0, HERE)
from compare_arms import permutation_test  # noqa: E402

# (kitchen, held-out group) -> metrics file. The unident_s_cx "mixed" group is
# the stage-1 MORL agents that learned mixed dishes, which no population used;
# it has no best responses, so only its raw return is reported. On unident_s_cx
# the 2m specialists are the full set (which free-rode) and the 5m ones partial.
SOURCES = {
    ("unident_s_cx", "onion soup (self-play held-out)"): "metrics_unident_s_cx_sp_phase4{budget}.json",
    ("unident_s_cx", "mixed dishes (MORL held-out)"): "metrics_unident_s_cx_morlho_phase4{budget}.json",
    ("pantry_cx", "mixed dishes (self-play held-out)"): "metrics_pantry_cx_sp_phase4{budget}.json",
}
# v2: the stable stage-2 rerun (schedules at the 2e6 pace, best checkpoint),
# scored against ZSC-Eval's HSP partners and, for continuity, the self-play set.
SOURCES_V2 = {
    ("unident_s_cx", "HSP partners"): "metrics_unident_s_cx_hsp_v2.json",
    ("unident_s_cx", "self-play held-out"): "metrics_unident_s_cx_sp_v2.json",
    ("pantry_cx", "HSP partners"): "metrics_pantry_cx_hsp_v2.json",
    ("pantry_cx", "self-play held-out"): "metrics_pantry_cx_sp_v2.json",
}
# A held-out partner counts only if a best response trained for it scored at
# least this much with it: a partner nobody can cook with tests nothing, and the
# rule does not look at any of the agents being compared.
COMPETENT = 50.0
LABELS = {
    "usual-hand": "Usual stage-2",
    "usual-dishann": "Usual stage-2, MORL ego",
    "partial-hand": "Specialists",
    "partial-dishann": "Specialists, MORL ego",
    "scripted-hand": "Specialists",
    "scripted-dishann": "Specialists, MORL ego",
    "bench_sp-cx-lr": "Stage-1 self-play",
    "bench_morl_ann-cx-lr": "Stage-1 MORL",
}
BASE = "usual-hand"


def arm_of(agent):
    return re.sub(r"-(5m|v2)$", "", re.sub(r"_s\d+$", "", re.sub(r"^s2_", "", agent)))


def partner_key(p):
    """heldout_hsp12_final -> 12 (as br_values keys HSP partners), heldout_sp7_mid -> sp7_mid."""
    m = re.search(r"hsp(\d+)_", p)
    return str(int(m.group(1))) if m else re.search(r"(sp\d+_\w+)$", p).group(1)


def scores(path):
    d = json.load(open(path))
    partners = set(d["partners"])
    cells = defaultdict(dict)
    for key, v in d["pair_matrix_stochastic"].items():
        a, b = key.split("|")
        cells[a][b] = v["mean"]
    agents = {a for a in cells if a not in partners} | {b for c in cells.values() for b in c if b not in partners}
    out = {}
    for a in agents:
        per = {}
        for p in partners:
            vals = [v for v in (cells.get(a, {}).get(p), cells.get(p, {}).get(a)) if v is not None]
            if vals:
                per[p] = float(np.mean(vals))
        if per:
            out[a] = per
    return out, sorted(partners)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--br", default=osp.join(RESULTS, "br_values_cx_best.json"))
    ap.add_argument("--budget", choices=["2m", "5m", "v2"], default="2m",
                    help="Stage-2 runs: 2m (the first), 5m (the -5m reruns, which collapse late) or v2 "
                    "(the stable rerun, scored against HSP partners too)")
    ap.add_argument("--br_hsp", default=osp.join(RESULTS, "br_values_cx_hsp.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    brs = json.load(open(args.br))
    if osp.exists(args.br_hsp):
        for layout, vals in json.load(open(args.br_hsp)).items():
            brs.setdefault(layout, {}).update(vals)
    args.out = args.out or osp.join(RESULTS, f"phase4_summary_{args.budget}.json")
    budget = {"5m": "_5m", "v2": "_v2"}.get(args.budget, "")
    summary = {}
    sources = SOURCES_V2 if args.budget == "v2" else SOURCES
    for (layout, group), fn in sources.items():
        path = osp.join(RESULTS, fn.format(budget=budget))
        if not osp.exists(path):
            print(f"== {layout} | {group}: {osp.basename(path)} not here yet")
            continue
        per_agent, partners = scores(path)
        br = {p: brs.get(layout, {}).get(partner_key(p), {}).get("last2") for p in partners} if "MORL" not in group else {}
        pool_best = {p: max(v[p] for v in per_agent.values() if p in v) for p in partners}
        denom = {p: max(br[p], pool_best[p]) for p in partners if br.get(p) is not None}
        if args.budget == "v2":
            dropped = sorted(p for p in denom if br[p] < COMPETENT)
            denom = {p: d for p, d in denom.items() if br[p] >= COMPETENT}
            partners = sorted(denom)
            per_agent = {a: {p: v for p, v in per.items() if p in denom} for a, per in per_agent.items()}
            if dropped:
                print(f"   ({len(dropped)} partners below the competence bar dropped: {dropped})")
        arms = defaultdict(dict)
        for a, per in per_agent.items():
            zs = float(np.mean(list(per.values())))
            bp = float(np.mean([per[p] / denom[p] for p in denom if p in per and denom[p] > 0])) if denom else None
            arms[arm_of(a)][a] = {"zero_shot": zs, "br_prox": bp}
        table = {}
        for arm, runs in arms.items():
            zs = [r["zero_shot"] for r in runs.values()]
            bp = [r["br_prox"] for r in runs.values() if r["br_prox"] is not None]
            table[arm] = {"label": LABELS.get(arm, arm), "n": len(runs), "zero_shot": float(np.mean(zs)),
                          "zero_shot_sd": float(np.std(zs, ddof=1)) if len(zs) > 1 else None,
                          "br_prox": float(np.mean(bp)) if bp else None, "runs": runs}
        tests = {}
        metric = "br_prox" if denom else "zero_shot"
        col = lambda arm: [r[metric] for r in arms[arm].values()]  # noqa: E731
        for arm in arms:
            if arm != BASE and BASE in arms:
                d, p, *_ = permutation_test(col(arm), col(BASE))
                tests[f"{arm} vs {BASE}"] = {"metric": metric, "diff": float(d), "p": float(p)}
        for morl, hand in (("partial-dishann", "partial-hand"), ("scripted-dishann", "scripted-hand"),
                           ("usual-dishann", "usual-hand"), ("bench_morl_ann-cx-lr", "bench_sp-cx-lr")):
            if morl in arms and hand in arms:
                d, p, *_ = permutation_test(col(morl), col(hand))
                tests[f"MORL: {morl} vs {hand}"] = {"metric": metric, "diff": float(d), "p": float(p)}
        summary[f"{layout} | {group}"] = {"partners": len(partners), "with_br": len(denom), "arms": table, "tests": tests}
        print(f"== {layout}: partners who cook {group}, {len(partners)} partners ({len(denom)} with a best response)")
        for arm, t in sorted(table.items(), key=lambda kv: -(kv[1]["br_prox"] or kv[1]["zero_shot"] / 1000)):
            bp = f"BR-Prox {t['br_prox']:.3f}" if t["br_prox"] is not None else ""
            print(f"  {t['label']:28s} ({arm:22s}) n={t['n']:2d}  zero-shot {t['zero_shot']:6.1f}  {bp}")
        for name, t in tests.items():
            print(f"    {name:48s} {t['metric']:9s} {t['diff']:+8.3f}  p={t['p']:.3f}")
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
