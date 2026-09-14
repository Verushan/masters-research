"""One-line-per-group view of a metrics_*.json from analyze_crossplay.

    python experiments/group_summary.py experiments/results/metrics_unident_s_hsp_live3.json
"""

import argparse
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+")
    args = parser.parse_args()
    for path in args.metrics:
        m = json.load(open(path))
        br = m.get("br_hat", {})
        print(f"\n=== {path}  (partner group {m['partner_group']}, pool-max BR mean {np.mean(list(br.values())):.1f})")
        print(f"{'group':22s} {'n':>2s} {'self-play':>14s} {'ZSC mean':>14s} {'worst':>7s} {'spread':>7s} {'stab':>6s} {'BR-prox':>8s}")
        for g, d in m["per_group"].items():
            sp, z = d["self_play"], d["zsc_mean"]
            brp = d.get("br_prox", {}).get("mean", float("nan"))
            print(
                f"{g:22s} {sp['n']:2d} {sp['mean']:7.1f}±{sp['std']:5.1f} {z['mean']:7.1f}±{z['std']:5.1f} "
                f"{d['zsc_worst']['mean']:7.1f} {d['zsc_spread']['mean']:7.1f} {d['return_stability']['mean']:6.1f} {brp:8.2f}"
            )


if __name__ == "__main__":
    main()
