"""Scripted reference agents for the fill-in suite: the ceiling for Step 6.

The go/no-go question is how much a partner-adaptive agent could gain over our
trained agents. A best response trained per scripted partner would answer it,
but the env only builds scripted partners in evaluation mode, so training one
needs plumbing. Two scripted agents answer it without training, and they are
scored by exactly the same harness and metrics as our agents:

generalist   ZSC-Eval's `place_onion_and_deliver_soup` script. It fills pots or
             serves, and switches whenever its current task is impossible --
             in effect a hand-coded fill-in rule, with no knowledge of the
             partner. If trained agents cannot fill in as well as this, the
             adaptation gap is real and large.
oracle       Knows the partner and plays its exact complement: a server beside
             the potter, a potter beside the server, the generalist beside
             anyone else, switching the moment the partner is swapped. What a
             perfectly partner-adaptive agent built from these scripts scores.

Both write records in the fillin_eval format, so analyze_fillin.py reads them
alongside the trained agents (entropy is 0: a script is deterministic given
its own internal state).

    python experiments/fillin_reference.py --layout unident_s --mode oracle \\
        --out experiments/results/fillin/unident_s/ref_oracle_s1.json.gz
"""

import argparse
import gzip
import json
import os
import os.path as osp
import pickle
import random
import sys
import zlib

import numpy as np
from loguru import logger

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "zsc-eval"))

from fillin_eval import PARTNERS, _new_script, idle_interacts, kitchen, move_target, step_events  # noqa: E402
from zsceval.envs.morl.tasks import TASKS, task_completions, task_demand  # noqa: E402
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action  # noqa: E402

POLICY_POOL = os.environ.get("POLICY_POOL")

# The complement of each partner, for the oracle.
COMPLEMENT = {
    "potter": "server",
    "server": "potter",
    "idle": "generalist",
    "clutter": "generalist",
    "generalist": "generalist",
    "dial2": "generalist",
    "dial5": "generalist",
    "dial8": "generalist",
    # Multi-recipe only: the cook who fetches the other ingredient and serves.
    "onion_potter": "tomato_cook",
    "tomato_potter": "onion_cook",
}
# Reference agents that are nobody's partner, for the oracle's complements.
NEW_REF_AGENTS = {
    "tomato_cook": dict(make=_new_script("menu_cook_tomato")),
    "onion_cook": dict(make=_new_script("menu_cook_onion")),
    "solo_cook": dict(make=_new_script("menu_cook_solo")),
    "solo_potter": dict(make=_new_script("menu_potter_solo")),
}
# The multi-recipe oracle differs in two places. Beside a potter it cooks as
# well as serving: with five-ingredient dishes the pot is the bottleneck, and the
# serve-only complement lost to the blind generalist (2.12 vs 2.75 soups/100 on
# unident_s_cx). Beside a server or an idle partner, who never pot, it makes
# only dishes it can finish alone, which in a split pantry means onion soup on
# its own side. Not beside the clutterer: the onions it leaves on the shared
# counter let a full cook finish mixed dishes (45 vs 20 reward/100 on pantry_cx).
NEW_COMPLEMENT = {
    **COMPLEMENT,
    "potter": "generalist",
    "server": "solo_potter",
    "idle": "solo_cook",
}


def run_episode(env, schedule, mode, seat, partners=None):
    partners = partners or PARTNERS
    new = partners is not PARTNERS
    agents = {**partners, **(NEW_REF_AGENTS if new else {})}
    complement = NEW_COMPLEMENT if new else COMPLEMENT
    obs, _share, available = env.reset()
    base, mdp = env.base_env, env.base_mdp
    partner_seat = 1 - seat
    segments = sorted(schedule)
    seg_i = 0
    agent = None

    def install(partner_name):
        nonlocal agent
        partner = partners[partner_name]["make"]()
        partner.reset(mdp, base.state, partner_seat)
        env.script_agent = [partner, None] if seat == 1 else [None, partner]
        # The generalist keeps its own state across a swap: it has no idea the
        # partner changed. The oracle is replaced with the new complement.
        if agent is None or mode == "oracle":
            agent = agents[complement[partner_name] if mode == "oracle" else "generalist"]["make"]()
            agent.reset(mdp, base.state, seat)

    install(segments[0][1])
    # sparse: the team's delivery reward this step. On the multi-recipe menus a
    # soup's worth varies (10-60), so soups/100 alone undercounts big dishes.
    keys = ("partner", "demand", "done", "action", "entropy", "blocked", "stay", "idle_interact", "sparse")
    rec = {k: [] for k in keys}
    totals = None
    for t in range(400):
        if seg_i + 1 < len(segments) and t >= segments[seg_i + 1][0]:
            seg_i += 1
            install(segments[seg_i][1])
        state = base.state
        demand = task_demand(mdp, state)
        # Step the agent's script here rather than installing it in the env, so
        # the action it chose is the one recorded.
        agent_action = agent.step(mdp, state, seat)
        a = Action.ACTION_TO_INDEX[agent_action]
        before = [pl.position for pl in state.players]
        joint = [[a], [4]] if seat == 0 else [[4], [a]]
        obs, _share, _rew, dones, info, available = env.step(np.array(joint))
        after = [pl.position for pl in base.state.players]

        target = move_target(before[seat], agent_action)
        blocked = bool(
            target is not None and after[seat] == before[seat] and target in (before[partner_seat], after[partner_seat])
        )
        shaped, totals = step_events(env, info, totals)
        done = task_completions(shaped, info["sparse_r_by_agent"])
        if seat == 1:
            done = done[::-1]
        rec["partner"].append(segments[seg_i][1])
        rec["demand"].append(demand.tolist())
        rec["done"].append(done.tolist())
        rec["action"].append(a)
        rec["entropy"].append(0.0)
        rec["blocked"].append(blocked)
        rec["stay"].append(agent_action == Action.STAY)
        rec["idle_interact"].append(idle_interacts(shaped[seat]))
        rec["sparse"].append(float(sum(info["sparse_r_by_agent"])))
        if np.all(dones):
            break
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--mode", choices=["generalist", "oracle"], required=True)
    ap.add_argument("--schedules", nargs="*", default=None, help="Default: every schedule the layout's env has")
    ap.add_argument("--seats", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert POLICY_POOL, "POLICY_POOL is unset"
    with open(osp.join(POLICY_POOL, args.layout, "policy_config", "mlp_policy_config.pkl"), "rb") as f:
        all_args = pickle.load(f)[0]
    all_args.layout_name = args.layout
    all_args.episode_length = 400
    Overcooked, partners, schedules = kitchen(args.layout)
    all_args.old_dynamics = partners is PARTNERS
    args.schedules = args.schedules or list(schedules)
    env = Overcooked(all_args, run_dir=osp.dirname(osp.abspath(__file__)), evaluation=True)

    name = f"ref_{args.mode}_s{args.seed + 1}"
    episodes = []
    for seat in args.seats:
        for sched in args.schedules:
            for ep in range(args.episodes):
                seed = args.seed * 100000 + seat * 50000 + zlib.crc32(sched.encode()) % 500 * 100 + ep
                random.seed(seed)
                np.random.seed(seed)
                rec = run_episode(env, schedules[sched], args.mode, seat, partners)
                episodes.append({"seat": seat, "schedule": sched, "episode": ep, "seed": seed, **rec})
            done = np.array(episodes[-1]["done"])
            logger.info(f"{name} seat {seat} vs {sched}: {int(done[:, :, 3].sum())} deliveries")

    payload = {
        "layout": args.layout,
        "agent": name,
        "actor": f"script:{args.mode}",
        "config": None,
        "env_flags": "",
        "tasks": list(TASKS),
        "partners": {k: {"covers": v["covers"]} for k, v in partners.items()},
        "schedules": {k: schedules[k] for k in args.schedules},
        "deterministic": False,
        "seats": args.seats,
        "episodes": episodes,
    }
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with gzip.open(args.out, "wt") as f:
        json.dump(payload, f, separators=(",", ":"))
    logger.success(f"wrote {len(episodes)} episodes to {args.out}")


if __name__ == "__main__":
    main()
