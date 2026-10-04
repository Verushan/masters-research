"""Does an agent know which partner it is playing with?

The specialist hypothesis is that an agent trained against scripted extremes
works out which kind of partner it is facing and plays for it. If so, the
partner's identity should be readable from the agent's recurrent state -- its
memory of the game so far -- and more readable than from the current screen
alone, which shows only where everyone stands right now.

For each agent: play every scripted partner from both seats for a few games,
record the recurrent state and the observation at every step, and fit a linear
(softmax) probe that names the partner. The probe is trained on some games and
scored on games it never saw, so it measures what generalises, not memorised
trajectories. Accuracy is reported over the game in windows of 50 steps:
recognition should start near chance and rise as evidence accumulates.

Three probes per agent:
  memory   the recurrent state (what the agent carries forward)
  screen   the current observation (the control: what anyone could see)
  both     memory and screen together
A single frame already says a lot -- what the partner holds, where it stands --
so the integration test is both > screen: what the agent's memory adds beyond
the current screen.

    python experiments/partner_probe.py --layout unident_s \\
        --out experiments/results/partner_probe_unident_s.json
"""

import argparse
import json
import os
import os.path as osp
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, osp.join(HERE, "..", "zsc-eval"))

PARTNERS = ["potter", "server", "idle", "clutter", "generalist", "dial2", "dial5", "dial8"]
WINDOW = 50

# (group, label, manifest folder, agent-name prefix). Stage-2 agents only: the
# stage-1 agents are feed-forward and carry no memory to probe.
GROUPS = [
    ("a", "A: specialists, with swaps", "{layout}-step8", "s2_scripted-hand_s"),
    ("ans", "A-ns: specialists, no swaps", "{layout}-step8", "s2_scripted-handns_s"),
    ("c2p", "C2′: corrected MORL", "{layout}-step8", "s2_scripted-noann2_s"),
    ("mixed", "Specialists + usual population, 22%", "{layout}-mixed", "s2_mixedspec-hand_s"),
    ("mix53", "Specialists + usual population, 53%", "{layout}-mixed", "s2_mixspec50-hand_s"),
    ("usual", "Usual stage-2", "{layout}-mixed", "s2_bench_sp_s"),
]


def manifest_rows(layout):
    rows = {}
    for _g, _l, folder, _p in GROUPS:
        path = osp.join(HERE, "results", "fillin", folder.format(layout=layout), "manifest.tsv")
        if not osp.exists(path):
            continue
        for line in open(path):
            parts = line.rstrip("\n").split("\t")
            rows.setdefault(parts[0], dict(actor=parts[1], config=parts[2], flags=parts[3] if len(parts) > 3 else ""))
    return rows


def record(args):
    """Play every partner from both seats; return memory, screen and labels per game."""
    name, row, layout, games, seed0 = args
    torch.set_num_threads(1)
    from fillin_eval import PARTNERS as SCRIPTS
    from fillin_eval import load_actor, parse_flags

    from zsceval.envs.overcooked.Overcooked_Env import Overcooked

    all_args, actor = load_actor(row["config"], row["actor"], parse_flags(row["flags"]))
    all_args.layout_name = layout
    all_args.episode_length = 400
    env = Overcooked(all_args, run_dir=HERE, evaluation=True)
    out = []
    for seat in (0, 1):
        for p_i, partner in enumerate(PARTNERS):
            for g in range(games):
                seed = seed0 + 1000 * seat + 37 * p_i + g
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                obs, _s, avail = env.reset()
                agent = SCRIPTS[partner]["make"]()
                agent.reset(env.base_mdp, env.base_env.state, 1 - seat)
                env.script_agent = [agent, None] if seat == 1 else [None, agent]
                rnn = np.zeros((1, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
                masks = np.ones((1, 1), dtype=np.float32)
                mem, scr = [], []
                for _t in range(all_args.episode_length):
                    o = np.asarray(obs[seat], dtype=np.float32)
                    with torch.no_grad():
                        probs, rnn_out = actor.get_probs(o[None], rnn, masks, np.asarray(avail[seat])[None])
                    rnn = rnn_out.cpu().numpy()
                    mem.append(rnn.reshape(-1).copy())
                    scr.append(o.reshape(-1) / 255.0)
                    p = probs.cpu().numpy().reshape(-1).astype(np.float64)
                    a = int(rng.choice(len(p), p=p / p.sum()))
                    joint = [[a], [4]] if seat == 0 else [[4], [a]]
                    obs, _s, _r, dones, _info, avail = env.step(np.array(joint))
                    if np.all(dones):
                        break
                m, sc = np.asarray(mem, dtype=np.float32), np.asarray(scr, dtype=np.float32)
                out.append({"seat": seat, "partner": p_i, "game": g, "memory": m, "screen": sc,
                            "both": np.concatenate([m, sc], 1)})
    return name, out


def fit_probe(X, y, n_classes, epochs=300, wd=1e-3):
    """Softmax regression with weight decay, full batch."""
    X = torch.tensor(X)
    y = torch.tensor(y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    W = torch.zeros(X.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=epochs, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, y) + wd * (W**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return lambda Z: ((torch.tensor(Z) - mu) / sd @ W + b).argmax(1).numpy()


def probe_agent(games, key, n_test):
    """Train on all but the last `n_test` games per (seat, partner); score per window."""
    train = [g for g in games if g["game"] < max(x["game"] for x in games) + 1 - n_test]
    test = [g for g in games if g not in train]
    Xtr = np.concatenate([g[key] for g in train])
    ytr = np.concatenate([np.full(len(g[key]), g["partner"]) for g in train])
    predict = fit_probe(Xtr, ytr, len(PARTNERS))
    acc = defaultdict(list)
    for g in test:
        pred = predict(g[key])
        for w in range(0, len(pred), WINDOW):
            acc[w].append(float((pred[w : w + WINDOW] == g["partner"]).mean()))
    windows = sorted(acc)
    return {"by_window": {int(w): float(np.mean(acc[w])) for w in windows},
            "overall": float(np.mean([np.mean(acc[w]) for w in windows]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", default="unident_s")
    ap.add_argument("--games", type=int, default=3, help="Games per (seat, partner); the last is held out")
    ap.add_argument("--per_group", type=int, default=5, help="Agents per group (runs 1..n)")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert os.environ.get("POLICY_POOL"), "POLICY_POOL is unset"

    rows = manifest_rows(args.layout)
    tasks, group_of = [], {}
    for g, _label, _f, prefix in GROUPS:
        for s in range(1, args.per_group + 1):
            name = f"{prefix}{s}"
            if name in rows:
                tasks.append((name, rows[name], args.layout, args.games, 7000))
                group_of[name] = g
    print(f"{len(tasks)} agents, {len(PARTNERS)} partners x 2 seats x {args.games} games each")

    results = {}
    with ProcessPoolExecutor(args.jobs) as ex:
        for name, games in ex.map(record, tasks):
            results[name] = {k: probe_agent(games, k, 1) for k in ("memory", "screen", "both")}
            r = results[name]
            print(f"  {name:28s} memory {r['memory']['overall']:.2f}  screen {r['screen']['overall']:.2f}  both {r['both']['overall']:.2f}", flush=True)

    summary = {}
    for g, label, *_ in GROUPS:
        names = [n for n in results if group_of[n] == g]
        if not names:
            continue
        summary[g] = {"label": label, "n": len(names)}
        for k in ("memory", "screen", "both"):
            ws = sorted(results[names[0]][k]["by_window"])
            summary[g][k] = {
                "overall": float(np.mean([results[n][k]["overall"] for n in names])),
                "by_window": {int(w): float(np.mean([results[n][k]["by_window"][w] for n in names])) for w in ws},
            }
    out = {"layout": args.layout, "chance": 1 / len(PARTNERS), "partners": PARTNERS, "window": WINDOW,
           "games_per_cell": args.games, "groups": summary, "agents": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\nchance = {1 / len(PARTNERS):.3f}")
    for g, v in summary.items():
        mw = " ".join(f"{v['memory']['by_window'][w]:.2f}" for w in sorted(v["memory"]["by_window"]))
        gain = v['both']['overall'] - v['screen']['overall']
        print(f"{v['label']:34s} n={v['n']}  memory {v['memory']['overall']:.2f}  screen {v['screen']['overall']:.2f}  both {v['both']['overall']:.2f}  (memory adds {gain:+.2f}) | memory by window: {mw}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
