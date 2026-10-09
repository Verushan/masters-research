"""Do the timed-order rules give partner adaptation room? Scripted pairs only.

The spec's validation bar (experiments/report/timed-orders-env-spec.md): a
queue-reading pair should keep expiries low, and an idle partner should make
them high. If the oracle cannot keep up the orders are too fast; if a lone cook
can, a partner does not matter and the env tests nothing about teamwork.

Pairs, both seats, N games each, 400 steps:
    oracle      order_cook + order_cook
    specialists order_potter + order_server (each does half the job)
    alone       order_cook + idle
    blind       the order-blind generalist script, both seats

    python experiments/timed_orders_check.py --layouts unident_s_m random1_m --games 10
"""

import argparse
import json
import os.path as osp
import sys

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(HERE, "..", "zsc-eval"))

from zsceval.envs.overcooked_new.script_agent.script_agent import SCRIPT_AGENTS  # noqa: E402
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld  # noqa: E402

PAIRS = {
    "oracle": ("order_cook", "order_cook"),
    "specialists": ("order_potter", "order_server"),
    "alone": ("order_cook", "idle"),
    "blind": ("place_onion_and_deliver_soup", "place_onion_and_deliver_soup"),
}


def play(layout, pair, seed, params, horizon=400):
    np.random.seed(seed)
    import random

    random.seed(seed)
    mdp = OvercookedGridworld.from_layout_name(layout, old_dynamics=False, timed_orders=params)
    state = mdp.get_standard_start_state()
    agents = [SCRIPT_AGENTS[n]() for n in pair]
    for i, a in enumerate(agents):
        a.reset(mdp, state, i)
    tot = dict(delivered=0, expired=0, unmatched=0, pay=0, sparse=0, arrived=1)
    for _ in range(horizon):
        joint = [a.step(mdp, state, i) for i, a in enumerate(agents)]
        n_before = len(state.orders)
        state, info = mdp.get_state_transition(state, joint)
        oi = info["order_info"]
        tot["delivered"] += sum(oi["delivered_by_agent"])
        tot["unmatched"] += sum(oi["unmatched_by_agent"])
        tot["pay"] += sum(oi["pay_by_agent"])
        tot["expired"] += oi["expired"]
        tot["sparse"] += sum(info["sparse_reward_by_agent"])
        tot["arrived"] += len(state.orders) - n_before + sum(oi["delivered_by_agent"]) + oi["expired"]
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layouts", nargs="+", default=["unident_s_m", "random1_m"])
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--arrival", type=int, default=30)
    ap.add_argument("--deadline", type=int, default=75)
    ap.add_argument("--queue", type=int, default=3)
    ap.add_argument("--penalty", type=int, default=10)
    ap.add_argument("--out")
    args = ap.parse_args()
    params = dict(queue=args.queue, arrival=args.arrival, deadline=args.deadline, penalty=args.penalty)
    print(f"params {params}")
    out = {"params": params, "results": {}}
    for layout in args.layouts:
        print(f"== {layout}")
        for name, pair in PAIRS.items():
            runs = []
            for g in range(args.games):
                seat_pair = pair if g % 2 == 0 else pair[::-1]
                runs.append(play(layout, seat_pair, g, params))
            m = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            closed = m["delivered"] + m["expired"]
            m["on_time_rate"] = m["delivered"] / closed if closed else 0.0
            out["results"][f"{layout}/{name}"] = m
            print(f"  {name:12s} return {m['sparse']:6.1f}  served {m['delivered']:4.1f}/{m['arrived']:4.1f} orders"
                  f"  expired {m['expired']:4.1f}  on-time {m['on_time_rate']:.2f}  unwanted soups {m['unmatched']:4.1f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
