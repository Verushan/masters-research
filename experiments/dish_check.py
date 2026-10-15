"""Do the complex-dish kitchens reward complex cooking? Scripted pairs only.

Phase 2a of the 15 Oct plan: kitchens whose menu pays more for bigger, mixed
dishes, so that what to cook is a real trade-off (speed vs value, onion vs
tomato trips) rather than "fill every pot with onions". Before any training is
spent on a menu, this prices each cooking strategy with scripted cooks that
can execute it perfectly:

    dish_X + dish_X     both cooks make only recipe X, in every pot
    valuable            order_cook_valuable: the most valuable menu dish first
    onion_script        the old order-blind onion script (fills pots with onions)
    alone               dish_<best> + idle: how much the second cook is worth

A menu is worth training on when (1) the best strategy clearly beats the
onion-only one, so complexity pays, (2) every listed strategy cooks at all
(no floor), and (3) one cook alone loses a good share of the return, so the
partner matters.

    python experiments/dish_check.py --layouts unident_s_cx random1_cx --games 6
"""

import argparse
import json
import os.path as osp
import random
import sys

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(HERE, "..", "zsc-eval"))

from zsceval.envs.overcooked_new.script_agent.script_agent import SCRIPT_AGENTS  # noqa: E402
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action  # noqa: E402
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (  # noqa: E402
    OvercookedGridworld,
    Recipe,
)


def code(recipe):
    return "".join("O" if i == "onion" else "T" for i in sorted(recipe.ingredients))


def play(layout, pair, seed, horizon=400):
    np.random.seed(seed)
    random.seed(seed)
    mdp = OvercookedGridworld.from_layout_name(layout, old_dynamics=False)
    state = mdp.get_standard_start_state()
    agents = [SCRIPT_AGENTS[n]() for n in pair]
    for i, a in enumerate(agents):
        a.reset(mdp, state, i)
    sparse, served = 0, {}
    for _ in range(horizon):
        joint = [a.step(mdp, state, i) for i, a in enumerate(agents)]
        prev = state
        state, info = mdp.get_state_transition(state, joint)
        r = sum(info["sparse_reward_by_agent"])
        sparse += r
        # A delivery is a soup leaving a player's hands at a window.
        for p0, p1 in zip(prev.players, state.players):
            if p0.has_object() and p0.get_object().name == "soup" and not p1.has_object():
                facing = Action.move_in_direction(p0.position, p0.orientation)
                if mdp.get_terrain_type_at_pos(facing) == "S":
                    k = code(p0.get_object().recipe)
                    served[k] = served.get(k, 0) + 1
    return sparse, served


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layouts", nargs="+", default=["unident_s_cx", "random1_cx"])
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--out")
    args = ap.parse_args()
    out = {}
    for layout in args.layouts:
        mdp = OvercookedGridworld.from_layout_name(layout, old_dynamics=False)
        menu = {code(r): (r.value, r.time) for r in map(Recipe.from_dict, mdp.start_all_orders)}
        print(f"== {layout}: menu " + ", ".join(f"{k} {v}/{t}t" for k, (v, t) in menu.items()))
        pairs = {f"dish_{k}": (f"dish_{k}", f"dish_{k}") for k in menu}
        pairs["valuable"] = ("order_cook_valuable", "order_cook_valuable")
        pairs["onion_script"] = ("place_onion_and_deliver_soup", "place_onion_and_deliver_soup")
        res = {}
        for name, pair in pairs.items():
            runs = [play(layout, pair if g % 2 == 0 else pair[::-1], g) for g in range(args.games)]
            res[name] = {"return": float(np.mean([r[0] for r in runs])),
                         "min": float(np.min([r[0] for r in runs])),
                         "served": {k: float(np.mean([r[1].get(k, 0) for r in runs]))
                                    for k in sorted({k for r in runs for k in r[1]})}}
        best = max((k for k in res if k.startswith("dish_")), key=lambda k: res[k]["return"])
        for name, pair in {"alone": (best, "idle")}.items():
            runs = [play(layout, pair if g % 2 == 0 else pair[::-1], g) for g in range(args.games)]
            res[name] = {"return": float(np.mean([r[0] for r in runs])), "min": float(np.min([r[0] for r in runs])),
                         "served": {}}
        for name, m in res.items():
            served = " ".join(f"{k}:{v:.1f}" for k, v in m["served"].items())
            print(f"  {name:14s} return {m['return']:6.1f} (min {m['min']:5.0f})  {served}")
        onion = max(res.get("dish_OOO", {"return": 0})["return"], res["onion_script"]["return"])
        print(f"  best fixed dish {best}: {res[best]['return']:.1f} = {res[best]['return'] / max(onion, 1e-9):.2f}x "
              f"the onion-only strategy; alone keeps {res['alone']['return'] / max(res[best]['return'], 1e-9):.0%}")
        out[layout] = {"menu": menu, "results": res, "best": best}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
