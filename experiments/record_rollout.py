"""Record one episode as a compact state trace, for animating in a report.

The upstream `render()` runs episodes and logs the return; it saves no frames.
This plays a frozen checkpoint against itself and writes what an animation
actually needs -- per step, where each cook is, which way it faces, what it is
carrying, and what is in each pot -- as JSON small enough to embed in a page.

Written for one specific comparison. `bench_morl_div` seed 6 on random0 scored
zero deliveries while driving the coordination objective to 317.8: it had found
that putting an item on the counter and taking it back scores a "handoff" every
time, forever. The same seed under the anchored objective delivers 196.7. Seeing
those two side by side is the clearest statement of the reward-design finding
that the project has, and neither a table nor a return curve conveys it.

    python experiments/record_rollout.py \
        --layout random0 \
        --actor random0/fcp/s1/bench_morl_div/sp6_final_actor.pt \
        --out experiments/results/trace_farming.json
"""

import argparse
import json
import os
import os.path as osp
import pickle
import sys

import numpy as np
import torch
from loguru import logger

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "..", "zsc-eval"))

from zsceval.algorithms.utils.util import init  # noqa: E402,F401  (torch init side effects)
from zsceval.envs.overcooked.Overcooked_Env import Overcooked  # noqa: E402
from zsceval.utils.util import get_shape_from_obs_space  # noqa: E402,F401

POLICY_POOL = os.environ.get("POLICY_POOL")


def held(player):
    """Name of whatever the cook is carrying, or None."""
    obj = getattr(player, "held_object", None)
    return None if obj is None else obj.name


def pot_states(state, mdp):
    """(position, how many ingredients, whether it is ready) for every pot."""
    out = []
    for pos in mdp.get_pot_locations():
        obj = state.objects.get(pos)
        if obj is None:
            out.append({"pos": list(pos), "n": 0, "ready": False})
            continue
        ingredients = getattr(obj, "ingredients", None)
        n = len(ingredients) if ingredients is not None else 0
        ready = bool(getattr(obj, "is_ready", False))
        out.append({"pos": list(pos), "n": n, "ready": ready})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--actor", required=True, help="Pool-relative actor checkpoint.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--config", default=None, help="Pool-relative policy config pickle.")
    args = ap.parse_args()

    assert POLICY_POOL, "POLICY_POOL is unset; source .env first"
    config_rel = args.config or f"{args.layout}/policy_config/mlp_policy_config.pkl"
    with open(osp.join(POLICY_POOL, config_rel), "rb") as f:
        all_args, obs_space, share_obs_space, act_space = pickle.load(f)

    # Rendering needs the raw env, not the vectorised wrapper, so the state is
    # reachable each step.
    all_args.layout_name = args.layout
    all_args.episode_length = args.steps
    env = Overcooked(all_args, run_dir=osp.dirname(osp.abspath(__file__)), evaluation=True)

    from zsceval.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor

    actor = R_Actor(all_args, obs_space, act_space, device=torch.device("cpu"))
    actor.load_state_dict(
        torch.load(osp.join(POLICY_POOL, args.actor), map_location="cpu")
    )
    actor.eval()

    obs, _, available = env.reset()
    base = env.base_env
    mdp = env.base_mdp
    frames, events = [], []
    rnn = np.zeros((2, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
    masks = np.ones((2, 1), dtype=np.float32)

    for t in range(args.steps):
        state = base.state
        frames.append(
            {
                "t": t,
                "cooks": [
                    {
                        "pos": list(p.position),
                        "dir": list(p.orientation),
                        "held": held(p),
                    }
                    for p in state.players
                ],
                "pots": pot_states(state, mdp),
            }
        )
        with torch.no_grad():
            action, _, rnn_out = actor(
                np.array(obs, dtype=np.float32),
                rnn,
                masks,
                np.array(available),
                deterministic=True,
            )
        rnn = rnn_out.detach().cpu().numpy()
        obs, _, rewards, dones, info, available = env.step(
            action.detach().cpu().numpy()
        )
        # Deliveries come from `sparse_r_by_agent`, not from the step reward:
        # the pool config carries reward_shaping_factor 1.0, so the step reward
        # also pays out for potting and plating, and counting it as deliveries
        # inflates the number six-fold.
        sparse = info.get("sparse_r_by_agent") if isinstance(info, dict) else None
        if sparse is not None:
            for i, v in enumerate(np.asarray(sparse, dtype=np.float64).ravel()):
                if v > 0:
                    events.append({"t": t, "cook": i, "kind": "delivery"})
        if np.all(dones):
            break

    payload = {
        "layout": args.layout,
        "grid": ["".join(r) for r in mdp.terrain_mtx],
        "actor": args.actor,
        "steps": len(frames),
        "deliveries": len(events),
        "frames": frames,
        "events": events,
    }
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    logger.success(
        f"{args.out}: {len(frames)} steps, {len(events)} deliveries, "
        f"{osp.getsize(args.out) / 1024:.0f} KB"
    )


if __name__ == "__main__":
    main()
