"""BR-Prox, ZSC-Eval's own zero-shot metric, for every agent type on unident_s and random1.

For each held-out partner, a best response (BR) was trained against that
partner alone (pipelines/br-partners.slurm, 2e6 steps). An agent's BR-Prox with
a partner is its return with that partner divided by the best return anyone
achieved with it: max(the BR's own evaluation, the best agent in our pool). The
agent's BR-Prox is the mean over the 16 partners; we also report the mean over
its worst quarter of partners (CVaR-25%), the robustness view ZSC-Eval uses.

Both sides are measured the same way: the BR is evaluated with sampled actions
from both seats, so the agents' returns come from the stochastic cross-play
pass, averaged over both seat orders. BRs trained at 2e6 steps are a lower bound
on the true best response, so this is "proximity to the best response we could
find" and errs on the generous side.

    python experiments/br_prox.py --out experiments/results/br_prox.json
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

SOURCES = {
    "unident_s": [
        "metrics_unident_s_hsp_scripted2.json",
        "metrics_unident_s_hsp_scripted.json",
        "metrics_unident_s_hsp_scripted_ns.json",
        "metrics_unident_s_hsp_mixed.json",
        "metrics_unident_s_s2_hsp_ego20.json",
        "metrics_unident_s_hsp_pool.json",
    ],
    "random1": ["metrics_random1_hsp_scripted.json", "metrics_random1_s2_hsp_ego12.json"],
    "random3": ["metrics_random3_hsp_scripted.json"],
}
LABELS = {
    "s2_scripted-hand": "A: specialists",
    "s2_scripted-handns": "A-ns: specialists, no swaps",
    "s2_scripted-noann2": "C2′: corrected MORL",
    "s2_scripted-neglect2": "C2: corrected MORL, fade",
    "s2_scripted-noann": "C′: original MORL",
    "s2_scripted-neglect": "C: original MORL, fade",
    "s2_mixedspec-hand": "Specialists + usual population",
    "s2_bench_sp": "Usual stage-2",
    "s2_bench_sp-annego": "Usual stage-2, MORL trainee",
    "bench_sp": "Stage-1 self-play",
    **{f"s2_scripted-hand-k{k}": f"A with {k} specialist{'s' if k > 1 else ''}" for k in (1, 2, 3, 4)},
    **{f"s2_scripted-noann2-k{k}": f"C2′ with {k} specialist{'s' if k > 1 else ''}" for k in (1, 2, 3, 4)},
}


def stochastic_cells(path):
    d = json.load(open(path))
    cells = {}
    for key, v in (d.get("pair_matrix_stochastic") or {}).items():
        a, b = key.split("|")
        cells[(a, b)] = v["mean"]
    return d["partners"], cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--br", nargs="+", default=[osp.join(RESULTS, "br_values_unident_s_random1.json"),
                                                 osp.join(RESULTS, "br_values_random3.json")])
    ap.add_argument("--br_value", choices=["final", "last2", "best"], default="last2")
    ap.add_argument("--out", default=osp.join(RESULTS, "br_prox.json"))
    args = ap.parse_args()
    brs = {}
    for path in args.br:
        if osp.exists(path):
            brs.update(json.load(open(path)))
    out = {}

    for layout, files in SOURCES.items():
        if layout not in brs:
            continue
        # agent -> partner index -> list of symmetrised stochastic returns
        score = defaultdict(lambda: defaultdict(list))
        for fn in files:
            path = osp.join(RESULTS, fn)
            if not osp.exists(path):
                continue
            partners, cells = stochastic_cells(path)
            agents = {a for pair in cells for a in pair if a not in partners}
            for a in agents:
                for p in partners:
                    vals = [cells[k] for k in ((a, p), (p, a)) if k in cells]
                    if vals:
                        score[a][int(re.search(r"hsp(\d+)", p).group(1))].append(float(np.mean(vals)))
        per_agent = {a: {p: float(np.mean(v)) for p, v in ps.items()} for a, ps in score.items()}
        br = {int(k): v[args.br_value] for k, v in brs[layout].items() if v.get(args.br_value) is not None}
        # Only partners with a trained BR: a pool-max stand-in can be near zero and
        # would inflate every ratio built on it.
        partners = sorted({p for ps in per_agent.values() for p in ps} & set(br))
        pool_best = {p: max(ps[p] for ps in per_agent.values() if p in ps) for p in partners}
        denom = {p: max(br.get(p, 0.0), pool_best[p]) for p in partners}

        def prox(a):
            r = np.array([per_agent[a][p] / denom[p] for p in partners if p in per_agent[a] and denom[p] > 0])
            worst = np.sort(r)[: max(1, len(r) // 4)]
            return float(r.mean()), float(worst.mean()), len(r)

        arms = defaultdict(list)
        for a in per_agent:
            arms[re.sub(r"_s\d+(?:r\d+)?$", "", a)].append(a)
        table = {}
        for arm, members in arms.items():
            vals = [prox(a) for a in members]
            table[arm] = {
                "label": LABELS.get(arm, arm),
                "n": len(members),
                "br_prox": float(np.mean([v[0] for v in vals])),
                "br_prox_sd": float(np.std([v[0] for v in vals], ddof=1)) if len(vals) > 1 else None,
                "cvar25": float(np.mean([v[1] for v in vals])),
                "runs": {a: v[0] for a, v in zip(members, vals)},
            }
        tests = {}
        if "s2_bench_sp" in table:
            ref = list(table["s2_bench_sp"]["runs"].values())
            for arm, t in table.items():
                if arm != "s2_bench_sp" and t["n"] > 1:
                    diff, p, *_ = permutation_test(list(t["runs"].values()), ref)
                    tests[arm] = {"diff_vs_usual": float(diff), "p": float(p)}
        out[layout] = {
            "denominator": {p: {"br": br.get(p), "pool_best": pool_best[p], "used": denom[p]} for p in partners},
            "br_beats_pool_on": int(sum(br.get(p, 0) >= pool_best[p] for p in partners)),
            "arms": table,
            "vs_usual_stage2": tests,
        }

        print(f"== {layout}: {len(partners)} partners; trained BR >= best pool agent on {out[layout]['br_beats_pool_on']}")
        for arm, t in sorted(table.items(), key=lambda kv: -kv[1]["br_prox"]):
            tt = tests.get(arm)
            extra = f"  vs usual {tt['diff_vs_usual']:+.3f} p={tt['p']:.3f}" if tt else ""
            print(f"  {t['label']:34s} n={t['n']:2d}  BR-Prox {t['br_prox']:.3f}  worst-25% {t['cvar25']:.3f}{extra}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
