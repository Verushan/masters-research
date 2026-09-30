"""Do agents of different types choose the same action in the same situation?

The behaviour profiles in behaviour_similarity.py compare what agents *end up
doing*. This compares their decisions directly: every agent is shown the same
game histories, and we record which action it would take at each step.

1. Reference games. One agent per type (its seed-1 run) plays every scripted
   partner from both seats. Both seats' actions are recorded, including the
   scripted partner's, so each game can be replayed move for move.
2. Replay. Each game is replayed once per observation encoding -- the MORL arms
   see extra weight channels, which depend on their own flags -- and the
   observations are stored. Replay feeds recorded actions to both seats, so the
   state sequence is identical whatever the encoding.
3. Decisions. Every agent runs its actor over every stored history, carrying its
   recurrent state, and outputs an action distribution per step.
4. Agreement between two agents is the share of steps where their most likely
   actions match, plus the mean Jensen-Shannon divergence of their
   distributions. Types are compared as in behaviour_similarity.py: within-type
   vs between-type, with a permutation test at the training-run level.

    python experiments/action_agreement.py --layout unident_s \\
        --out experiments/results/action_agreement_unident_s.json
"""

import argparse
import itertools
import json
import os
import os.path as osp
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from loguru import logger

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, osp.join(HERE, "..", "zsc-eval"))

from behaviour_similarity import GROUPS, TESTS, group_of, perm_test  # noqa: E402
from fillin_eval import PARTNERS, load_actor, parse_flags  # noqa: E402

from zsceval.envs.overcooked.Overcooked_Env import Overcooked  # noqa: E402
from zsceval.envs.overcooked.script_agent.base import BaseScriptAgent  # noqa: E402

POLICY_POOL = os.environ.get("POLICY_POOL")
RESULTS = osp.join(HERE, "results")
SINGLE = ["potter", "server", "idle", "clutter", "generalist", "dial2", "dial5", "dial8"]


class Recorder(BaseScriptAgent):
    """A scripted partner that remembers every move it made."""

    def __init__(self, inner, log):
        super().__init__()
        self.inner, self.log = inner, log

    def reset(self, mdp, state, player_idx):
        self.inner.reset(mdp, state, player_idx)

    def step(self, mdp, state, player_idx):
        a = self.inner.step(mdp, state, player_idx)
        self.log.append(a)
        return a


class Replayer(BaseScriptAgent):
    """Plays back a recorded partner, move for move."""

    def __init__(self, moves):
        super().__init__()
        self.moves, self.t = moves, 0

    def reset(self, mdp, state, player_idx):
        self.t = 0

    def step(self, mdp, state, player_idx):
        a = self.moves[self.t]
        self.t += 1
        return a


def manifest(layout):
    rows = {}
    for folder in (f"{layout}-step8", layout):
        path = osp.join(RESULTS, "fillin", folder, "manifest.tsv")
        for line in open(path):
            parts = line.rstrip("\n").split("\t")
            name, actor, config = parts[:3]
            flags = parts[3] if len(parts) > 3 else ""
            rows.setdefault(name, dict(actor=actor, config=config, flags=flags))
    return rows


def make_env(layout, config, flags):
    all_args, actor = None, None
    with open(osp.join(POLICY_POOL, config), "rb") as f:
        import pickle

        all_args = pickle.load(f)[0]
    for k, v in parse_flags(flags).items():
        setattr(all_args, k, v)
    all_args.layout_name = layout
    all_args.episode_length = 400
    env = Overcooked(all_args, run_dir=HERE, evaluation=True)
    assert env.agent_idx == 0
    return env, all_args


def get_probs(actor, obs, rnn, masks, avail):
    with torch.no_grad():
        probs, rnn_out = actor.get_probs(obs, rnn, masks, avail)
    return probs.cpu().numpy().astype(np.float64), rnn_out.cpu().numpy()


def record_game(env, actor, all_args, partner, seat, seed):
    """One stochastic game of `actor` in `seat`; returns both seats' moves."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs, _share, avail = env.reset()
    log = []
    rec = Recorder(PARTNERS[partner]["make"](), log)
    rec.reset(env.base_mdp, env.base_env.state, 1 - seat)
    env.script_agent = [rec, None] if seat == 1 else [None, rec]
    rnn = np.zeros((1, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
    masks = np.ones((1, 1), dtype=np.float32)
    agent_moves = []
    for _ in range(all_args.episode_length):
        p, rnn = get_probs(actor, np.asarray(obs[seat], dtype=np.float32)[None], rnn, masks, np.asarray(avail[seat])[None])
        p = p.reshape(-1) / p.sum()
        a = int(rng.choice(len(p), p=p))
        agent_moves.append(a)
        joint = [[a], [4]] if seat == 0 else [[4], [a]]
        obs, _share, _r, dones, _info, avail = env.step(np.array(joint))
        if np.all(dones):
            break
    return {"partner": partner, "seat": seat, "agent": agent_moves, "partner_moves": log}


def replay_obs(env, game):
    """Replay a recorded game; the observation and action mask seen from its seat."""
    seat = game["seat"]
    obs, _share, avail = env.reset()
    rp = Replayer(game["partner_moves"])
    rp.reset(env.base_mdp, env.base_env.state, 1 - seat)
    env.script_agent = [rp, None] if seat == 1 else [None, rp]
    O, A = [], []
    for a in game["agent"]:
        O.append(np.asarray(obs[seat], dtype=np.float32))
        A.append(np.asarray(avail[seat]))
        joint = [[a], [4]] if seat == 0 else [[4], [a]]
        obs, _share, _r, _d, _info, avail = env.step(np.array(joint))
    return np.stack(O), np.stack(A)


def decisions(actor, all_args, O, A):
    """Action distributions of `actor` along every stored history, batched over games."""
    G, T = O.shape[:2]
    rnn = np.zeros((G, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
    masks = np.ones((G, 1), dtype=np.float32)
    out = np.zeros((G, T, 6))
    for t in range(T):
        p, rnn = get_probs(actor, O[:, t], rnn, masks, A[:, t])
        out[:, t] = p / p.sum(-1, keepdims=True)
    return out


def js(P, Q):
    M = 0.5 * (P + Q)

    def kl(X, Y):
        return np.where(X > 0, X * (np.log(np.maximum(X, 1e-12)) - np.log(np.maximum(Y, 1e-12))), 0).sum(-1)

    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", default="unident_s")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert POLICY_POOL, "POLICY_POOL is unset; source .env first"

    rows = manifest(args.layout)
    agents = sorted(
        (n for n in rows if group_of(n) not in (None, "oracle")),
        key=lambda n: ([g for g, *_ in GROUPS].index(group_of(n)), int(n.rsplit("_s", 1)[1])),
    )
    logger.info(f"{len(agents)} agents")

    # 1. Reference games: each type's seed-1 agent, every partner, both seats.
    games = []
    for g, *_ in GROUPS:
        ref = next((a for a in agents if group_of(a) == g and a.endswith("_s1")), None)
        if ref is None:
            continue
        r = rows[ref]
        env, all_args = make_env(args.layout, r["config"], r["flags"])
        _args, actor = load_actor(r["config"], r["actor"], parse_flags(r["flags"]))
        for seat in (0, 1):
            for i, partner in enumerate(SINGLE):
                games.append({**record_game(env, actor, _args, partner, seat, seed=1000 * seat + 17 * i + len(games)), "by": ref})
        logger.info(f"recorded games by {ref}")
    T = min(len(gm["agent"]) for gm in games)

    # 2. One observation bank per distinct encoding.
    banks = {}
    for a in agents:
        key = (rows[a]["config"], rows[a]["flags"])
        if key in banks:
            continue
        env, _ = make_env(args.layout, *key)
        O, A = zip(*(replay_obs(env, gm) for gm in games))
        banks[key] = (np.stack([o[:T] for o in O]), np.stack([m[:T] for m in A]))
        logger.info(f"bank {key[0].rsplit('/', 1)[-1]} {key[1] or '(base)'}: {banks[key][0].shape}")

    # 3. Every agent's decisions on every history.
    P = []
    for a in agents:
        r = rows[a]
        all_args, actor = load_actor(r["config"], r["actor"], parse_flags(r["flags"]))
        O, A = banks[(r["config"], r["flags"])]
        P.append(decisions(actor, all_args, O, A).reshape(-1, 6))
    P = np.stack(P)  # agents x states x 6
    logger.info(f"decisions: {P.shape}")

    # 4. Pairwise agreement.
    n = len(agents)
    top = P.argmax(-1)
    agree = np.zeros((n, n))
    jsd = np.zeros((n, n))
    for i in range(n):
        agree[i] = (top == top[i]).mean(1)
        jsd[i] = js(P[i][None], P).mean(1)
    groups = [group_of(a) for a in agents]
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    order = [g for g, *_ in GROUPS if g in idx]

    def block(M, ga, gb):
        b = M[np.ix_(idx[ga], idx[gb])]
        return b[np.triu_indices(len(idx[ga]), 1)] if ga == gb else b.ravel()

    mat = {f"{ga}|{gb}": {"agree": float(block(agree, ga, gb).mean()), "js": float(block(jsd, ga, gb).mean())} for ga in order for gb in order}
    tests = []
    for ga, gb in TESTS:
        if ga in idx and gb in idx:
            diff, p, exact = perm_test(jsd, idx[ga], idx[gb])
            tests.append({"a": ga, "b": gb, "js_between_minus_within": diff, "p": p, "exact": exact,
                          "agree_within_a": mat[f"{ga}|{ga}"]["agree"], "agree_within_b": mat[f"{gb}|{gb}"]["agree"],
                          "agree_between": mat[f"{ga}|{gb}"]["agree"]})

    labels = {k: lab for k, lab, *_ in GROUPS}
    out = {
        "layout": args.layout,
        "games": len(games),
        "steps_per_game": int(T),
        "states": int(P.shape[1]),
        "reference_agents": sorted({gm["by"] for gm in games}),
        "groups": [{"key": g, "label": labels[g], "n": len(idx[g])} for g in order],
        "matrix": mat,
        "tests": tests,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"{len(games)} games x {T} steps = {P.shape[1]} shared states, {n} agents")
    print("\nsame top action, share of states (within type on the diagonal)")
    print(" " * 10 + "".join(f"{g:>9s}" for g in order))
    for ga in order:
        print(f"{ga:>10s}" + "".join(f"{mat[f'{ga}|{gb}']['agree']:9.2f}" for gb in order))
    print("\nbetween vs within (JS divergence; permutation test, run as unit)")
    for t in tests:
        print(f"  {t['a']:>8s} vs {t['b']:<8s} agree within {t['agree_within_a']:.2f}/{t['agree_within_b']:.2f} "
              f"between {t['agree_between']:.2f}  JS diff {t['js_between_minus_within']:+.4f}  p={t['p']:.4f}{'' if t['exact'] else ' (sampled)'}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
