"""Training-time summary of every arm in a training-curves JSON.

Final self-play return per seed and, for adaptive arms, where the preference
vector ended up. This is the part of the analysis that does not need the
cross-play: whether an arm trains at all, how seed-stable it is, and whether
the mirror-descent update did what it was meant to.

    python experiments/summarize_training.py \
        experiments/results/training_curves_unident_s_live3.json
"""

import argparse
import json

import numpy as np


def tail(series, n=10):
    return float(np.mean(series["value"][-n:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("curves", nargs="+")
    parser.add_argument("--tail", type=int, default=10, help="logged points averaged")
    parser.add_argument("--out", default=None, help="JSON summary")
    args = parser.parse_args()

    summary = {}
    for path in args.curves:
        d = json.load(open(path))
        print(f"\n=== {path}")
        print(f"{'arm':26s} {'n':>2s} {'sparse':>7s} {'sd':>6s}   per-seed (last {args.tail} points)")
        summary[path] = {}
        for arm, seeds in d.items():
            rows = sorted(seeds.items(), key=lambda kv: int(kv[0]))
            vals = [tail(r["series"]["ep_sparse_r"], args.tail) for _, r in rows if "ep_sparse_r" in r["series"]]
            if not vals:
                continue
            v = np.array(vals)
            sd = v.std(ddof=1) if len(v) > 1 else float("nan")
            print(f"{arm:26s} {len(v):2d} {v.mean():7.1f} {sd:6.1f}   {' '.join(f'{x:5.1f}' for x in v)}")
            entry = {"n": len(v), "sparse_mean": float(v.mean()), "sparse_sd": float(sd), "per_seed": vals}
            wkeys = [k for k in rows[0][1]["series"] if k.startswith("ep_morl_w_")]
            if wkeys:
                w = {
                    k[len("ep_morl_w_"):]: [tail(r["series"][k], args.tail) for _, r in rows if k in r["series"]]
                    for k in wkeys
                }
                entry["final_w"] = w
                print(
                    f"{'':26s}    final w  "
                    + "  ".join(f"{k}={np.mean(x):.3f}±{np.std(x):.3f}" for k, x in w.items())
                )
            okeys = [k for k in rows[0][1]["series"] if k.startswith("ep_obj_")]
            if okeys:
                o = {
                    k[len("ep_obj_"):]: float(np.mean([tail(r["series"][k], args.tail) for _, r in rows if k in r["series"]]))
                    for k in okeys
                }
                entry["final_obj"] = o
                print(f"{'':26s}    objectives  " + "  ".join(f"{k}={x:.1f}" for k, x in o.items()))
            summary[path][arm] = entry
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
