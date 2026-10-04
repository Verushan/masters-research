"""Sanity-check a specialist family before training on it.

Each member plays a few games beside the scripted generalist, from both seats,
on each layout. Reported per member: soups the pair delivers, the share of the
pair's pot-filling the member did (does its task-mix knob do what it says), and
how often it stood still (does its laziness knob). A member that never lets the
pair cook is useless as a training partner on that layout -- the random3 lesson.

    python experiments/family_check.py --layouts unident_s random1 --k 16 \\
        --out experiments/results/family_check.json
"""

import argparse
import json
import os
import os.path as osp
import pickle
import random
import sys

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, osp.join(HERE, "..", "zsc-eval"))

from zsceval.envs.morl.tasks import task_completions  # noqa: E402
from zsceval.envs.overcooked.Overcooked_Env import Overcooked  # noqa: E402
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action  # noqa: E402
from zsceval.envs.overcooked.script_agent import family  # noqa: E402
from zsceval.envs.overcooked.script_agent.script_agent import SCRIPT_AGENTS  # noqa: E402

PARTNER = "place_onion_and_deliver_soup"


def make_env(layout):
    pool = os.environ["POLICY_POOL"]
    with open(osp.join(pool, layout, "policy_config", "mlp_policy_config.pkl"), "rb") as f:
        all_args = pickle.load(f)[0]
    all_args.layout_name = layout
    all_args.episode_length = 400
    return Overcooked(all_args, run_dir=HERE, evaluation=True)


def play(env, member, seat, seed):
    random.seed(seed)
    np.random.seed(seed)
    env.reset()
    agents = [None, None]
    agents[seat] = SCRIPT_AGENTS[member]()
    agents[1 - seat] = SCRIPT_AGENTS[PARTNER]()
    for a, ag in enumerate(agents):
        ag.reset(env.base_mdp, env.base_env.state, a)
    env.script_agent = agents
    fills = np.zeros(2)
    soups, stay = 0, 0
    for _ in range(400):
        before = env.base_env.state.players[seat].position
        _o, _s, _r, dones, info, _a = env.step(np.array([[4], [4]]))
        done = task_completions(info["shaped_info_by_agent"], info["sparse_r_by_agent"])
        fills += np.asarray(done)[:, 0]
        soups += int(np.asarray(done)[:, 3].sum())
        stay += int(env.base_env.state.players[seat].position == before)
        if np.all(dones):
            break
    share = fills[seat] / fills.sum() if fills.sum() else float("nan")
    return soups, share, stay / 400


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layouts", nargs="+", default=["unident_s", "random1"])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    members = family.sample(args.k)
    out = {"members": members, "partner": PARTNER, "layouts": {}}
    for layout in args.layouts:
        env = make_env(layout)
        rows = {}
        for m in members:
            res = [play(env, m, seat, 100 * g + seat) for seat in (0, 1) for g in range(args.games)]
            s, sh, st = zip(*res)
            rows[m] = {"soups": float(np.mean(s)), "fill_share": float(np.nanmean(sh)) if not all(np.isnan(sh)) else None,
                       "still": float(np.mean(st)), "min_soups": int(min(s))}
        out["layouts"][layout] = rows
        print(f"== {layout} (member beside the scripted generalist, {2 * args.games} games)")
        print(f"   {'member':18s} {'soups':>6s} {'min':>4s} {'fill share':>10s} {'still':>6s}")
        for m, r in rows.items():
            fs = "  n/a" if r["fill_share"] is None else f"{r['fill_share']:.2f}"
            print(f"   {m:18s} {r['soups']:6.1f} {r['min_soups']:4d} {fs:>10s} {r['still']:6.2f}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
