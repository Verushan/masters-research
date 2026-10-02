"""The tests pre-registered in experiments/report/preregistration-2026-10-07.md.

1. Partner removal (unident_s): A and C2' trained against K = 1..5 of the five
   specialists. Zero-shot score per run from metrics_unident_s_hsp_pool.json.
     1. slope of A over K (permute K labels across A's runs)
     2. slope(C2') - slope(A) (permute the arm label within each (K, seed) pair)
     3. A(K) against C2'(K = 5), for each K < 5 (descriptive)
2. MORL reward on the stage-2 trainee, over three kitchens:
     - mean: per-kitchen difference scaled by the pooled SD, averaged over
       kitchens, arm labels permuted within kitchen
     - spread: log(SD_MORL / SD_hand) averaged over kitchens, same permutation;
       plus the direction per kitchen
     - replication of the unident_s spread lead on seeds 13-20 alone
     - kitchens where either arm's mean is below 20 are flagged as floored

Every permutation test is two-sided, with 20,000 samples (or exact when small).

    python experiments/prereg_tests.py --out experiments/results/prereg_tests.json
"""

import argparse
import itertools
import json
import os.path as osp
import re

import numpy as np

RESULTS = osp.join(osp.dirname(osp.abspath(__file__)), "results")
N_PERM = 20000
FLOOR = 20.0


def runs(metrics, arm):
    """{seed: zero-shot mean} for an arm, from analyze_crossplay output."""
    rx = re.compile(rf"^{re.escape(arm)}_s(\d+)$")
    out = {}
    for agent, v in metrics["per_agent"].items():
        m = rx.match(agent)
        if m and v.get("zsc_mean") is not None:
            out[int(m.group(1))] = v["zsc_mean"]
    return out


def slope(k, y):
    k, y = np.asarray(k, float), np.asarray(y, float)
    return float(np.polyfit(k, y, 1)[0])


def two_sided(null, observed):
    null = np.asarray(null)
    return float((np.abs(null) >= abs(observed) - 1e-12).mean())


def partner_removal(rng):
    d = json.load(open(osp.join(RESULTS, "metrics_unident_s_hsp_pool.json")))
    data = {}
    for arm, name in (("a", "s2_scripted-hand"), ("c2p", "s2_scripted-noann2")):
        for k in (1, 2, 3, 4):
            data[(arm, k)] = runs(d, f"{name}-k{k}")
        data[(arm, 5)] = runs(d, name)
    table = {
        f"{arm}|{k}": {"n": len(v), "mean": float(np.mean(list(v.values()))), "sd": float(np.std(list(v.values()), ddof=1)),
                       "runs": v}
        for (arm, k), v in data.items()
    }

    # 1. Slope of A over K.
    ka = [k for k in range(1, 6) for _ in data[("a", k)]]
    ya = [v for k in range(1, 6) for v in data[("a", k)].values()]
    obs_a = slope(ka, ya)
    null = [slope(rng.permutation(ka), ya) for _ in range(N_PERM)]
    t1 = {"slope_a_per_partner": obs_a, "p": two_sided(null, obs_a)}
    kc = [k for k in range(1, 6) for _ in data[("c2p", k)]]
    yc = [v for k in range(1, 6) for v in data[("c2p", k)].values()]
    obs_c = slope(kc, yc)
    null = [slope(rng.permutation(kc), yc) for _ in range(N_PERM)]
    t1c = {"slope_c2p_per_partner": obs_c, "p": two_sided(null, obs_c)}

    # 2. Slope difference, permuting the arm label within (K, seed) pairs.
    pairs = []
    for k in range(1, 6):
        for s in sorted(set(data[("a", k)]) & set(data[("c2p", k)])):
            pairs.append((k, data[("a", k)][s], data[("c2p", k)][s]))
    K = np.array([p[0] for p in pairs], float)
    A = np.array([p[1] for p in pairs])
    C = np.array([p[2] for p in pairs])
    obs = slope(K, C) - slope(K, A)
    null = []
    for _ in range(N_PERM):
        flip = rng.random(len(pairs)) < 0.5
        a2, c2 = np.where(flip, C, A), np.where(flip, A, C)
        null.append(slope(K, c2) - slope(K, a2))
    t2 = {"slope_c2p_minus_slope_a": obs, "p": two_sided(null, obs), "n_pairs": len(pairs)}

    # 3. A(K) against C2'(K = 5).
    from compare_arms import permutation_test  # noqa: E402

    t3 = []
    ref = list(data[("c2p", 5)].values())
    for k in (1, 2, 3, 4):
        diff, p, *_ = permutation_test(list(data[("a", k)].values()), ref)
        t3.append({"k": k, "a_minus_c2p_k5": float(diff), "p": float(p)})
    return {"table": table, "test1_a": t1, "test1_c2p": t1c, "test2": t2, "test3": t3}


def stage2(rng):
    kitchens = {
        "unident_s": "metrics_unident_s_s2_hsp_ego20.json",
        "random0": "metrics_random0_s2_hsp_ego12.json",
        "random1": "metrics_random1_s2_hsp_ego12.json",
    }
    per = {}
    for kit, fn in kitchens.items():
        d = json.load(open(osp.join(RESULTS, fn)))
        h, m = runs(d, "s2_bench_sp"), runs(d, "s2_bench_sp-annego")
        per[kit] = {"hand": h, "morl": m}

    def summary(h, m):
        h, m = np.array(list(h.values())), np.array(list(m.values()))
        pooled = np.sqrt(((len(h) - 1) * h.var(ddof=1) + (len(m) - 1) * m.var(ddof=1)) / (len(h) + len(m) - 2))
        return {
            "n_hand": len(h), "n_morl": len(m),
            "mean_hand": float(h.mean()), "mean_morl": float(m.mean()),
            "sd_hand": float(h.std(ddof=1)), "sd_morl": float(m.std(ddof=1)),
            "worst_hand": float(h.min()), "worst_morl": float(m.min()),
            "d": float((m.mean() - h.mean()) / pooled),
            "log_sd_ratio": float(np.log(m.std(ddof=1) / h.std(ddof=1))),
            "floored": bool(min(h.mean(), m.mean()) < FLOOR),
        }

    out = {k: summary(v["hand"], v["morl"]) for k, v in per.items()}

    def pooled_test(kits):
        obs_d = np.mean([out[k]["d"] for k in kits])
        obs_s = np.mean([out[k]["log_sd_ratio"] for k in kits])
        nd, ns = [], []
        arrays = {k: np.concatenate([list(per[k]["hand"].values()), list(per[k]["morl"].values())]) for k in kits}
        for _ in range(N_PERM):
            ds, ss = [], []
            for k in kits:
                x = rng.permutation(arrays[k])
                nh = len(per[k]["hand"])
                h, m = x[:nh], x[nh:]
                pooled = np.sqrt(((len(h) - 1) * h.var(ddof=1) + (len(m) - 1) * m.var(ddof=1)) / (len(h) + len(m) - 2))
                ds.append((m.mean() - h.mean()) / pooled)
                ss.append(np.log(m.std(ddof=1) / h.std(ddof=1)))
            nd.append(np.mean(ds))
            ns.append(np.mean(ss))
        return {
            "kitchens": kits,
            "mean_effect_d": float(obs_d), "p_mean": two_sided(nd, obs_d),
            "log_sd_ratio": float(obs_s), "sd_ratio": float(np.exp(obs_s)), "p_spread": two_sided(ns, obs_s),
            "spread_direction_lower_for_morl": sum(out[k]["log_sd_ratio"] < 0 for k in kits),
        }

    all_k = list(kitchens)
    unfloored = [k for k in all_k if not out[k]["floored"]]
    pooled = {"all": pooled_test(all_k), "unfloored": pooled_test(unfloored)}

    # Replication on the new unident_s seeds alone.
    h = np.array([v for s, v in per["unident_s"]["hand"].items() if s >= 13])
    m = np.array([v for s, v in per["unident_s"]["morl"].items() if s >= 13])
    obs = np.log(m.std(ddof=1) / h.std(ddof=1))
    x = np.concatenate([h, m])
    null = []
    for _ in range(N_PERM):
        y = rng.permutation(x)
        null.append(np.log(y[len(h):].std(ddof=1) / y[: len(h)].std(ddof=1)))
    rep = {"n": [len(h), len(m)], "sd_hand": float(h.std(ddof=1)), "sd_morl": float(m.std(ddof=1)),
           "mean_hand": float(h.mean()), "mean_morl": float(m.mean()),
           "log_sd_ratio": float(obs), "p": two_sided(null, obs), "replicates_direction": bool(obs < 0)}
    return {"per_kitchen": out, "pooled": pooled, "replication_unident_s_13_20": rep}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=osp.join(RESULTS, "prereg_tests.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    import sys

    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
    pr = partner_removal(rng)
    s2 = stage2(rng)
    with open(args.out, "w") as f:
        json.dump({"partner_removal": pr, "stage2": s2}, f, indent=1)

    print("== 1. Partner removal (unident_s, zero-shot vs 16 HSP)")
    for arm in ("a", "c2p"):
        print(f"  {arm:4s} " + "  ".join(f"K{k}: {pr['table'][f'{arm}|{k}']['mean']:6.1f} (n={pr['table'][f'{arm}|{k}']['n']})" for k in range(1, 6)))
    print(f"  Q1 slope A  {pr['test1_a']['slope_a_per_partner']:+.1f}/partner  p={pr['test1_a']['p']:.4f}")
    print(f"     slope C2' {pr['test1_c2p']['slope_c2p_per_partner']:+.1f}/partner  p={pr['test1_c2p']['p']:.4f}")
    print(f"  Q2 slope(C2') - slope(A) {pr['test2']['slope_c2p_minus_slope_a']:+.2f}  p={pr['test2']['p']:.4f}  ({pr['test2']['n_pairs']} pairs)")
    for t in pr["test3"]:
        print(f"  Q3 A(K={t['k']}) - C2'(K=5) {t['a_minus_c2p_k5']:+.1f}  p={t['p']:.4f}")
    print("\n== 2. Stage-2 MORL trainee")
    for k, v in s2["per_kitchen"].items():
        print(f"  {k:10s} hand {v['mean_hand']:6.1f} sd {v['sd_hand']:5.1f} | MORL {v['mean_morl']:6.1f} sd {v['sd_morl']:5.1f} | d {v['d']:+.2f}  sd ratio {np.exp(v['log_sd_ratio']):.2f}{'  FLOORED' if v['floored'] else ''}")
    for name, p in s2["pooled"].items():
        print(f"  pooled ({name}: {', '.join(p['kitchens'])}): mean d {p['mean_effect_d']:+.2f} p={p['p_mean']:.3f} | sd ratio {p['sd_ratio']:.2f} p={p['p_spread']:.3f} | lower spread in {p['spread_direction_lower_for_morl']}/{len(p['kitchens'])}")
    r = s2["replication_unident_s_13_20"]
    print(f"  replication, unident_s seeds 13-20: sd hand {r['sd_hand']:.1f} vs MORL {r['sd_morl']:.1f} (means {r['mean_hand']:.1f} / {r['mean_morl']:.1f}), p={r['p']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
