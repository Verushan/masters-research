"""Which dishes get cooked before and after a mid-game partner swap.

Reads fill-in records (fillin_eval.py / fillin_reference.py, which store the
recipe of every delivery per step on the multi-recipe kitchens) and, for every
swap schedule, splits each game at its swaps. Per agent type and segment:
dishes and points per 100 steps, and the dish mix -- the share of big mixed
dishes (5 ingredients), medium (4), onion soup, small, and off-menu dishes.
Run-level means (the training run is the unit), then pooled per agent type.

    python experiments/dish_swaps.py experiments/results/fillin/pantry_cx-swapdish/*.json.gz \\
        --out experiments/results/dish_swaps_pantry_cx.json
"""

import argparse
import gzip
import json
import re
from collections import defaultdict

import numpy as np

CLASSES = ("big mixed", "medium mixed", "onion soup", "small", "off-menu")


def dish_class(code, menu):
    if code not in menu:
        return "off-menu"
    if len(code) == 5:
        return "big mixed"
    if len(code) == 4:
        return "medium mixed"
    if code == "OOO":
        return "onion soup"
    return "small"


def segments(partners):
    out, start = [], 0
    for t in range(1, len(partners) + 1):
        if t == len(partners) or partners[t] != partners[start]:
            out.append((partners[start], start, t))
            start = t
    return out


def arm_of(agent):
    return re.sub(r"_s\d+$", "", agent)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

    runs = defaultdict(lambda: defaultdict(list))  # agent -> (schedule, segment index) -> segment stats
    layout = None
    for path in args.files:
        d = json.load(gzip.open(path, "rt"))
        layout = d["layout"]
        for ep in d["episodes"]:
            if "dishes" not in ep:
                continue
            segs = segments(ep["partner"])
            if len(segs) < 2:
                continue
            for i, (partner, a, b) in enumerate(segs):
                dishes = [c for step in ep["dishes"][a:b] for c in step]
                runs[d["agent"]][(ep["schedule"], i, partner)].append(
                    {"steps": b - a, "dishes": dishes, "points": float(np.sum(ep["sparse"][a:b]))})
    mdp = OvercookedGridworld.from_layout_name(layout, old_dynamics=False)
    menu = {"".join("O" if i == "onion" else "T" for i in sorted(o["ingredients"])) for o in mdp.start_all_orders}

    def summarise(segs):
        steps = sum(s["steps"] for s in segs)
        dishes = [c for s in segs for c in s["dishes"]]
        mix = {k: 0 for k in CLASSES}
        for c in dishes:
            mix[dish_class(c, menu)] += 1
        n = max(1, len(dishes))
        return {"dishes_per_100": 100 * len(dishes) / steps, "points_per_100": 100 * sum(s["points"] for s in segs) / steps,
                **{f"share_{k}": v / n for k, v in mix.items()}}

    per_run = {agent: {f"{k[0]}|{k[1]}|{k[2]}": summarise(v) for k, v in segs.items()} for agent, segs in runs.items()}
    arms = defaultdict(lambda: defaultdict(list))
    for agent, keys in per_run.items():
        for key, stats in keys.items():
            arms[arm_of(agent)][key].append(stats)
    table = {arm: {key: {m: float(np.mean([s[m] for s in v])) for m in v[0]} | {"n": len(v)} for key, v in keys.items()}
             for arm, keys in arms.items()}
    with open(args.out, "w") as f:
        json.dump({"layout": layout, "menu": sorted(menu), "arms": table, "runs": per_run}, f, indent=1)

    schedules = sorted({k.split("|")[0] for keys in table.values() for k in keys})
    for sched in schedules:
        print(f"\n== {layout}: {sched}  (per segment: points/100 steps, dishes/100, share big mixed / onion soup / off-menu)")
        for arm in sorted(table):
            cells = [(k, v) for k, v in table[arm].items() if k.startswith(sched + "|")]
            cells.sort(key=lambda kv: int(kv[0].split("|")[1]))
            row = "   ".join(f"{k.split('|')[2]:>13s}: {v['points_per_100']:5.1f} pts {v['dishes_per_100']:4.2f} d "
                           f"{v['share_big mixed']:.0%}/{v['share_onion soup']:.0%}/{v['share_off-menu']:.0%}"
                           for k, v in cells)
            print(f"  {arm:26s} {row}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
