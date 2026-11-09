"""Summarise dish_trajectory.py output: which dishes each agent type converges on.

Per kitchen and agent type, the mean number of dishes per game of each class
(big mixed = 5 ingredients, medium mixed = 4, onion soup, small, off-menu) in
bins of training steps, the off-menu share of all deliveries, and per run:
the dish it settles on (most delivered at its last checkpoint) and the first
checkpoint at which off-menu dishes fall below 10% of deliveries for good.

    python experiments/analyze_dish_traj.py experiments/results/dish_traj/*.json \\
        --out experiments/results/dish_traj_summary.json
"""

import argparse
import json
import os.path as osp
from collections import Counter, defaultdict

import numpy as np

CLASSES = ("big mixed", "medium mixed", "onion soup", "small", "off-menu")
BIN = 0.5e6


def dish_class(code, menu):
    if code not in menu:
        return "off-menu"
    return {5: "big mixed", 4: "medium mixed"}.get(len(code), "onion soup" if code == "OOO" else "small")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {}
    for path in sorted(args.files):
        d = json.load(open(path))
        menu = d["menu"]
        key = f"{d['layout']} | {d['exp']}"
        bins = defaultdict(lambda: defaultdict(list))
        settle, clean_at = Counter(), []
        for seed, rows in d["runs"].items():
            for r in rows:
                b = int(r["step"] // BIN)
                per = defaultdict(float)
                for c, v in r["per_game"].items():
                    per[dish_class(c, menu)] += v
                for k in CLASSES:
                    bins[b][k].append(per[k])
                bins[b]["points"].append(r["points_per_game"])
                bins[b]["off_share"].append(r["off_menu_share"])
            last = rows[-1]["per_game"]
            settle[dish_class(max(last, key=last.get), menu) if last else "nothing"] += 1
            dirty = [i for i, r in enumerate(rows) if r["off_menu_share"] >= 0.10 or not r["per_game"]]
            first_clean = dirty[-1] + 1 if dirty else 0
            clean_at.append(rows[first_clean]["step"] if first_clean < len(rows) else None)
        table = {int(b): {k: float(np.mean(v)) for k, v in m.items()} for b, m in sorted(bins.items())}
        cleaned = [c for c in clean_at if c is not None]
        out[key] = {"runs": len(d["runs"]), "by_bin": table, "settles_on": dict(settle),
                    "runs_clean_by_end": len(cleaned), "median_clean_step": float(np.median(cleaned)) if cleaned else None}
        print(f"\n== {key}: {len(d['runs'])} runs; settle on {dict(settle)}; off-menu under 10% for good in "
              f"{len(cleaned)} runs (median from {np.median(cleaned) / 1e6 if cleaned else float('nan'):.1f}M)")
        print("   steps(M)  points  " + "  ".join(f"{k[:10]:>10s}" for k in CLASSES) + "   off-menu share")
        for b, m in table.items():
            print(f"   {b * BIN / 1e6:4.1f}-{(b + 1) * BIN / 1e6:3.1f}  {m['points']:6.1f}  "
                  + "  ".join(f"{m[k]:10.2f}" for k in CLASSES) + f"   {m['off_share']:.2f}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
