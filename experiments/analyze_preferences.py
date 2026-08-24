#!/usr/bin/env python
"""What the mirror-descent preference update actually did during training.

Reads `training_curves.json` and answers three questions that the cross-play
matrix cannot:

1. **Did the weights move at all?** `ep_morl_w_*` is the preference vector at the
   *end* of an episode. `MirrorDescentPreferences.reset()` runs on every
   `env.reset()`, so w restarts from the target each episode and this is the net
   within-episode drift -- which is exactly the "real time" adaptation the
   proposal describes, not a slow drift across training.

2. **Did they move the way the rule says?** The update pushes weight *away* from
   whatever is over-represented, so w_i and the realised share g_i should be
   anti-correlated across training. A near-zero correlation means the update is
   firing but not tracking anything.

3. **Did adapting help balance behaviour?** The stated goal is to keep the agent
   attending to whatever it is neglecting. If that works, the adaptive arm's
   realised shares should sit closer to uniform than the fixed-weight arm's.
   `imbalance` is the proposal's own `s = Var(g) / Var_max` in [0, 1]: 0 is a
   perfectly even split across objectives, 1 is one objective taking everything.

Usage:
    python experiments/analyze_preferences.py \
        --curves results/training_curves_random0.json --out results/preferences_random0.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

OBJECTIVES = ["task_completion", "ingredient_prep", "plating", "coordination"]


def aligned(series, keys):
    """Stack `keys` from one run's series onto their common steps.

    Returns (steps, values) with values shaped (T, len(keys)), or (None, None)
    if any key is missing.
    """
    if any(k not in series for k in keys):
        return None, None
    common = set(series[keys[0]]["step"])
    for k in keys[1:]:
        common &= set(series[k]["step"])
    steps = sorted(common)
    if not steps:
        return None, None
    values = np.stack(
        [
            [dict(zip(series[k]["step"], series[k]["value"]))[s] for s in steps]
            for k in keys
        ],
        axis=-1,
    )
    return np.array(steps), values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    curves = json.load(open(args.curves))
    num_objectives = len(OBJECTIVES)
    var_max = (num_objectives - 1) / num_objectives**2

    result = {}
    for arm, runs in curves.items():
        per_seed = {}
        for seed, run in runs.items():
            series = run["series"]
            entry = {}

            steps, obj = aligned(series, [f"ep_obj_{o}" for o in OBJECTIVES])
            if steps is not None:
                totals = obj.sum(axis=-1, keepdims=True)
                shares = np.divide(
                    obj, totals, out=np.full_like(obj, 1.0 / num_objectives), where=totals > 0
                )
                imbalance = shares.var(axis=-1) / var_max
                entry["steps"] = steps.tolist()
                entry["shares"] = {o: shares[:, i].tolist() for i, o in enumerate(OBJECTIVES)}
                entry["imbalance"] = imbalance.tolist()
                entry["final_shares"] = {
                    o: float(np.mean(shares[-5:, i])) for i, o in enumerate(OBJECTIVES)
                }
                entry["final_imbalance"] = float(np.mean(imbalance[-5:]))
                entry["mean_imbalance"] = float(np.mean(imbalance))

            w_steps, weights = aligned(series, [f"ep_morl_w_{o}" for o in OBJECTIVES])
            # Drift is measured against the 1/K the update resets to, so it only
            # means anything for weights that live on the simplex. bench_sparse
            # uses w = (20,0,0,0) -- a scalarization, not a preference vector --
            # and would otherwise report a drift of 20.5 that describes nothing.
            on_simplex = w_steps is not None and np.allclose(weights.sum(axis=-1), 1.0, atol=1e-6)
            if w_steps is not None and not on_simplex:
                entry["weights_off_simplex"] = True
                entry["fixed_weights"] = {
                    o: float(weights[0, i]) for i, o in enumerate(OBJECTIVES)
                }
            if on_simplex:
                entry["weight_steps"] = w_steps.tolist()
                entry["weights"] = {o: weights[:, i].tolist() for i, o in enumerate(OBJECTIVES)}
                entry["final_weights"] = {
                    o: float(np.mean(weights[-5:, i])) for i, o in enumerate(OBJECTIVES)
                }
                # How far the end-of-episode weights drift from the 1/K they are
                # reset to. Zero means the update never bit.
                entry["weight_drift"] = float(
                    np.mean(np.abs(weights - 1.0 / num_objectives).sum(axis=-1))
                )
                entry["max_weight_drift"] = float(
                    np.max(np.abs(weights - 1.0 / num_objectives).sum(axis=-1))
                )
                if steps is not None:
                    common = sorted(set(steps.tolist()) & set(w_steps.tolist()))
                    if len(common) > 2:
                        s_idx = [steps.tolist().index(s) for s in common]
                        w_idx = [w_steps.tolist().index(s) for s in common]
                        correlations = {}
                        for i, objective in enumerate(OBJECTIVES):
                            g = shares[s_idx, i]
                            w = weights[w_idx, i]
                            if g.std() > 1e-12 and w.std() > 1e-12:
                                correlations[objective] = float(np.corrcoef(g, w)[0, 1])
                        entry["weight_share_correlation"] = correlations
            per_seed[seed] = entry

        summary = {}
        for field in ["final_imbalance", "mean_imbalance", "weight_drift", "max_weight_drift"]:
            values = [e[field] for e in per_seed.values() if field in e]
            if values:
                summary[field] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
        for field in ["final_shares", "final_weights"]:
            profiles = [e[field] for e in per_seed.values() if field in e]
            if profiles:
                summary[field] = {
                    o: float(np.mean([p[o] for p in profiles])) for o in OBJECTIVES
                }
        correlations = [
            e["weight_share_correlation"] for e in per_seed.values() if "weight_share_correlation" in e
        ]
        if correlations:
            summary["weight_share_correlation"] = {
                o: float(np.mean([c[o] for c in correlations if o in c]))
                for o in OBJECTIVES
                if any(o in c for c in correlations)
            }
        result[arm] = {"per_seed": per_seed, "summary": summary}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f)

    def fmt(v, nd=3):
        return "  -  " if v is None else f"{v:.{nd}f}"

    print(f"\n{'arm':16} {'imbalance s':>12} {'w drift':>9} {'max w drift':>12}   realised objective shares")
    print("-" * 100)
    for arm, blob in result.items():
        s = blob["summary"]
        shares = s.get("final_shares", {})
        share_str = " ".join(f"{o[:5]}={fmt(shares.get(o), 2)}" for o in OBJECTIVES)
        print(
            f"{arm:16} {fmt((s.get('final_imbalance') or {}).get('mean')):>12} "
            f"{fmt((s.get('weight_drift') or {}).get('mean')):>9} "
            f"{fmt((s.get('max_weight_drift') or {}).get('mean')):>12}   {share_str}"
        )

    for arm, blob in result.items():
        corr = blob["summary"].get("weight_share_correlation")
        if corr:
            print(f"\n{arm} corr(w_i, g_i) across training: " + ", ".join(f"{o}={v:+.2f}" for o, v in corr.items()))
        weights = blob["summary"].get("final_weights")
        if weights:
            print(f"{arm} end-of-episode weights: " + ", ".join(f"{o}={v:.3f}" for o, v in weights.items()))

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
