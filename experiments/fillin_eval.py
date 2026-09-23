"""Fill-in evaluation: play one of our agents against partners who neglect tasks.

Steps 2 and 3 of the Fill-In Evaluation Suite. Our 16 held-out partners never
demanded different responses -- best responses trained against each of them all
learned the same strategy -- so they cannot tell an adaptive agent from one that
simply does everything. Scripted partners can: each covers a known set of tasks
and neglects the rest, and swapping one for another mid-episode changes which
tasks are left undone while the agent is playing.

The agent sits in seat 0 and acts from its own actor; the partner in seat 1 is a
scripted policy the env steps itself (`Overcooked.script_agent`), so a swap is a
single assignment between steps. Every step records, from the state *before*
the joint action, which tasks the kitchen needed (`zsceval.envs.morl.tasks`),
and from the step's info which tasks each seat completed -- plus the agent's
full action distribution, so its entropy ("variance of actions") can be read
per partner. `experiments/analyze_fillin.py` turns the records into metrics.

    python experiments/fillin_eval.py --layout unident_s \\
        --agent bench_sp_s1=unident_s/fcp/s1/bench_sp/sp1_final_actor.pt \\
        --out experiments/results/fillin/unident_s_bench_sp_s1.json.gz

Widened-observation agents (partner picture, preference weights) need their
own policy config and the env flags they trained under:

    --config unident_s/policy_config/mlp_policy_config_mow-anchored_live3_mos.pkl \\
    --env_flags "use_morl=1 morl_objectives=anchored_live3 morl_weights=20,3,3 ..."
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
import torch
from loguru import logger

# insert, not append: .env exports PYTHONPATH, which may point at a different
# checkout of zsc-eval, and the harness must run the code that sits beside it.
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "zsc-eval"))

from zsceval.envs.morl.tasks import TASKS, task_completions, task_demand  # noqa: E402
from zsceval.envs.overcooked.Overcooked_Env import Overcooked  # noqa: E402
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action  # noqa: E402
from zsceval.envs.overcooked.script_agent.base import BaseScriptAgent  # noqa: E402
from zsceval.envs.overcooked.script_agent.script_agent import SCRIPT_AGENTS  # noqa: E402

POLICY_POOL = os.environ.get("POLICY_POOL")


class IdleAgent(BaseScriptAgent):
    """A partner who does nothing. The agent has to run the whole kitchen."""

    def step(self, mdp, state, player_idx):
        return Action.STAY


# Partner personalities. `covers` is what each is scripted to do; the analysis
# also measures what each actually did, since a script can stall.
PARTNERS = {
    "potter": dict(make=lambda: SCRIPT_AGENTS["place_onion_in_pot"](), covers=["fill_pot"]),
    "server": dict(
        make=lambda: SCRIPT_AGENTS["deliver_soup"](),
        covers=["fetch_dish", "plate_soup", "deliver"],
    ),
    "generalist": dict(make=lambda: SCRIPT_AGENTS["place_onion_and_deliver_soup"](), covers=list(TASKS)),
    "idle": dict(make=IdleAgent, covers=[]),
    # Picks up onions and dumps them on counters: busy, useless, in the way.
    "clutter": dict(make=lambda: SCRIPT_AGENTS["put_onion_everywhere"](), covers=[]),
    # The dial: a preference, not a specialism. It prefers onions k times in
    # ten, but switches task itself when its preferred one is impossible (no
    # room in a pot, no soup to take). The extremes (0 and 10) divide by zero
    # inside the script, which is why the pure specialists above exist.
    **{
        f"dial{k}": dict(
            make=(lambda k=k: SCRIPT_AGENTS[f"{k}onion_{10 - k}soup_0noise"]()),
            covers=list(TASKS),
        )
        for k in (2, 5, 8)
    },
}

# Swap schedules: (first step, partner). A single entry is a partner held for
# the whole episode. The two flips swap exact complements, so the task the
# agent should be doing reverses at the swap.
SCHEDULES = {
    **{name: [(0, name)] for name in PARTNERS},
    "potter>server": [(0, "potter"), (200, "server")],
    "server>potter": [(0, "server"), (200, "potter")],
    "potter>server>idle": [(0, "potter"), (133, "server"), (266, "idle")],
    # Step 8 held-out sequences: swap steps the Step 7 arms never saw, one with
    # their own partners in a new order, one with partners they never met.
    "server>idle>potter": [(0, "server"), (133, "idle"), (266, "potter")],
    "dial5>clutter>generalist": [(0, "dial5"), (133, "clutter"), (266, "generalist")],
}


def load_actor(config_rel, actor_rel, env_overrides):
    with open(osp.join(POLICY_POOL, config_rel), "rb") as f:
        all_args, obs_space, _share_obs_space, act_space = pickle.load(f)
    for k, v in env_overrides.items():
        setattr(all_args, k, v)
    from zsceval.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor

    actor = R_Actor(all_args, obs_space, act_space, device=torch.device("cpu"))
    actor.load_state_dict(torch.load(osp.join(POLICY_POOL, actor_rel), map_location="cpu"))
    actor.eval()
    return all_args, actor


def parse_flags(text):
    """'use_morl=true morl_weights=20,3,3 eta=0.5' -> typed overrides for all_args."""
    out = {}
    for item in (text or "").split():
        k, v = item.split("=", 1)
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                pass
        else:
            out[k] = v
    return out


def move_target(pos, action):
    if action in (Action.STAY, Action.INTERACT):
        return None
    return (pos[0] + action[0], pos[1] + action[1])


def run_episode(env, actor, all_args, schedule, deterministic, rng, seat=0):
    """One episode with the agent in base-env `seat`; returns per-step arrays.

    Every record is in agent-first order whichever seat it held: `done[t][0]`
    is the agent's completions, `done[t][1]` the partner's.
    """
    partner_seat = 1 - seat
    obs, _share, available = env.reset()
    base, mdp = env.base_env, env.base_mdp
    segments = sorted(schedule)
    seg_i = 0

    def install(name):
        agent = PARTNERS[name]["make"]()
        agent.reset(mdp, base.state, partner_seat)
        env.script_agent = [agent, None] if seat == 1 else [None, agent]

    install(segments[0][1])
    rnn = np.zeros((1, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
    masks = np.ones((1, 1), dtype=np.float32)
    rec = {k: [] for k in ("partner", "demand", "done", "action", "entropy", "blocked", "stay", "idle_interact")}

    for t in range(all_args.episode_length):
        if seg_i + 1 < len(segments) and t >= segments[seg_i + 1][0]:
            seg_i += 1
            install(segments[seg_i][1])
        state = base.state
        demand = task_demand(mdp, state)

        with torch.no_grad():
            probs, rnn_out = actor.get_probs(
                np.asarray(obs[seat], dtype=np.float32)[None],
                rnn,
                masks,
                np.asarray(available[seat])[None],
            )
        rnn = rnn_out.detach().cpu().numpy()
        p = probs.detach().cpu().numpy().reshape(-1).astype(np.float64)
        p = p / p.sum()
        a = int(np.argmax(p)) if deterministic else int(rng.choice(len(p), p=p))
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())

        before = [pl.position for pl in state.players]
        # The partner's index is ignored: the env replaces it with the script's move.
        joint = [[a], [4]] if seat == 0 else [[4], [a]]
        obs, _share, _rew, dones, info, available = env.step(np.array(joint))
        after = [pl.position for pl in base.state.players]

        agent_action = Action.INDEX_TO_ACTION[a]
        target = move_target(before[seat], agent_action)
        blocked = bool(
            target is not None
            and after[seat] == before[seat]
            and target in (before[partner_seat], after[partner_seat])
        )
        shaped = info["shaped_info_by_agent"]
        done = task_completions(shaped, info["sparse_r_by_agent"])
        if seat == 1:
            done = done[::-1]

        rec["partner"].append(segments[seg_i][1])
        rec["demand"].append(demand.tolist())
        rec["done"].append(done.tolist())
        rec["action"].append(a)
        rec["entropy"].append(entropy)
        rec["blocked"].append(blocked)
        rec["stay"].append(agent_action == Action.STAY)
        rec["idle_interact"].append(
            int(shaped[seat].get("IDLE_INTERACT_X", 0) + shaped[seat].get("IDLE_INTERACT_EMPTY", 0))
        )
        if np.all(dones):
            break
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--agent", required=True, help="NAME=pool-relative actor checkpoint")
    ap.add_argument("--config", default=None, help="Pool-relative policy config (default: the layout's mlp config)")
    ap.add_argument("--env_flags", default="", help="Space-separated k=v env/args overrides")
    ap.add_argument("--schedules", nargs="*", default=list(SCHEDULES))
    ap.add_argument(
        "--seats",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Base-env seats the agent plays from. Self-play agents often split roles by "
        "starting position, so an agent seen from one seat alone can look like a specialist.",
    )
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert POLICY_POOL, "POLICY_POOL is unset; source .env first"
    name, actor_rel = args.agent.split("=", 1)
    config_rel = args.config or f"{args.layout}/policy_config/mlp_policy_config.pkl"
    overrides = parse_flags(args.env_flags)
    all_args, actor = load_actor(config_rel, actor_rel, overrides)
    all_args.layout_name = args.layout
    all_args.episode_length = 400
    env = Overcooked(all_args, run_dir=osp.dirname(osp.abspath(__file__)), evaluation=True)
    assert env.agent_idx == 0, "the agent must hold base-env seat 0 for per-agent counters to line up"

    episodes = []
    for seat in args.seats:
        for sched in args.schedules:
            for ep in range(args.episodes):
                # crc32, not hash(): str hashes are salted per process, so hash()
                # seeds would differ between runs and episodes would not reproduce.
                seed = args.seed * 100000 + seat * 50000 + zlib.crc32(sched.encode()) % 500 * 100 + ep
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                rec = run_episode(env, actor, all_args, SCHEDULES[sched], args.deterministic, rng, seat)
                episodes.append({"seat": seat, "schedule": sched, "episode": ep, "seed": seed, **rec})
            done = np.array(episodes[-1]["done"])
            logger.info(
                f"{name} seat {seat} vs {sched}: last episode {int(done[:, :, 3].sum())} deliveries, "
                f"agent tasks {done[:, 0].sum(0).tolist()} partner tasks {done[:, 1].sum(0).tolist()}"
            )

    payload = {
        "layout": args.layout,
        "agent": name,
        "actor": actor_rel,
        "config": config_rel,
        "env_flags": args.env_flags,
        "tasks": list(TASKS),
        "partners": {k: {"covers": v["covers"]} for k, v in PARTNERS.items()},
        "schedules": {k: SCHEDULES[k] for k in args.schedules},
        "deterministic": args.deterministic,
        "seats": args.seats,
        "episodes": episodes,
    }
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with gzip.open(args.out, "wt") as f:
        json.dump(payload, f, separators=(",", ":"))
    logger.success(f"wrote {len(episodes)} episodes to {args.out}")


if __name__ == "__main__":
    main()
