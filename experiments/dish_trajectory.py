"""Which dishes agents cook over the course of training, on the complex-dish kitchens.

Replays checkpoints saved during training (W&B run files, every 25 updates in
stage 1 and every 50 in the stable stage-2 runs) and records the recipe of
every soup delivered, off-menu ones included. Per checkpoint: deliveries per
game of each recipe, the share of off-menu dishes, and the points they earned.
That shows whether runs converge on a dish, which, and how fast the worthless
(off-menu) dishes die out.

Partners:
    self         stage 1: the checkpoint plays itself (both seats)
    scripts      stage 2: the ego (seat 0, then seat 1) beside each scripted
                 training partner in turn (--scripts, the specialist set)
    yml          stage 2: beside each frozen entry of its training population
                 yml in turn (--yml, pool-relative: the usual population)

Sampled actions throughout, as every zero-shot number in this project.

    python experiments/dish_trajectory.py --layout pantry_cx --exp bench_sp-cx-lr --partner self \\
        --out experiments/results/dish_traj_pantry_cx_bench_sp.json
    python experiments/dish_trajectory.py --layout pantry_cx --exp fcp-S2-scripted-dishann-v2 --partner scripts \\
        --scripts potter server idle dial2 dial8 onion_potter tomato_potter --out ...
"""

import argparse
import json
import os
import os.path as osp
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, osp.join(HERE, "..", "zsc-eval"))

import render_gameplay as rg  # noqa: E402  (Seat, make_env, make_script)

from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp import overcooked_mdp as new_mdp  # noqa: E402

DELIVERED = []


def _record(orig):
    def deliver_soup(self, state, player, soup):
        DELIVERED.append(tuple(sorted(soup.ingredients)))
        return orig(self, state, player, soup)

    return deliver_soup


new_mdp.OvercookedGridworld.deliver_soup = _record(new_mdp.OvercookedGridworld.deliver_soup)


def code(ingredients):
    return "".join("O" if i == "onion" else "T" for i in sorted(ingredients))


def checkpoints(run, prefix, n):
    files = [f for f in run.files() if f.name.startswith(prefix + "actor_periodic_")]
    by_step = sorted((int(f.name.split("_")[-1].split(".pt")[0]), f) for f in files)
    if len(by_step) <= n:
        return by_step
    idx = np.unique(np.linspace(0, len(by_step) - 1, n).round().astype(int))
    return [by_step[i] for i in idx]


def play(layout, ego, partner, seed, ego_seat):
    env = rg.make_env(layout)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    seats = [ego, partner] if ego_seat == 0 else [partner, ego]
    scripts = [None, None]
    obs, _share, avail = env.reset()
    for i, s in enumerate(seats):
        s.reset()
        if s.kind == "script":
            scripts[i] = rg.make_script(s.name, layout)
            scripts[i].reset(env.base_mdp, env.base_env.state, i)
    env.script_agent = scripts
    DELIVERED.clear()
    points = 0.0
    for _t in range(400):
        joint = [[s.act(obs[i], avail[i], False, rng)] if s.kind == "actor" else [4] for i, s in enumerate(seats)]
        obs, _share, _r, dones, info, avail = env.step(np.array(joint))
        points += float(sum(info["sparse_r_by_agent"]))
        if np.all(dones):
            break
    return Counter(code(d) for d in DELIVERED), points


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--partner", choices=["self", "scripts", "yml"], required=True)
    ap.add_argument("--scripts", nargs="*", default=[])
    ap.add_argument("--yml", default=None, help="Pool-relative population yml (partner=yml)")
    ap.add_argument("--yml_every", type=int, default=3, help="Use every k-th frozen entry of the yml")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="Training seeds (default: every finished run)")
    ap.add_argument("--n_ckpt", type=int, default=12, help="Checkpoints per run, evenly spaced")
    ap.add_argument("--episodes", type=int, default=4, help="Games per checkpoint (per partner, for scripts)")
    ap.add_argument("--project", default="Overcooked-new")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import wandb

    pool = os.environ["POLICY_POOL"]
    api = wandb.Api(timeout=120)
    runs = [r for r in api.runs(f"{os.environ['WANDB_ENTITY']}/{args.project}",
                                filters={"config.experiment_name": args.exp, "config.layout_name": args.layout,
                                         "state": "finished"})
            if args.seeds is None or r.config["seed"] in args.seeds]
    stage2 = args.partner in ("scripts", "yml")
    frozen = []
    if args.partner == "yml":
        import yaml

        pop = yaml.safe_load(open(osp.join(pool, args.yml)))
        frozen = [(v["policy_config_path"], v["model_path"]["actor"]) for v in pop.values()
                  if not v.get("train") and "model_path" in v][:: args.yml_every]
        print(f"population: {len(frozen)} partners from {args.yml}")
    prefix, config = ("fcp_adaptive/", rg.RNN) if stage2 else ("", rg.MLP)
    mdp = new_mdp.OvercookedGridworld.from_layout_name(args.layout, old_dynamics=False)
    menu = {code(o["ingredients"]): new_mdp.Recipe.from_dict(o).value for o in mdp.start_all_orders}
    out = {"layout": args.layout, "exp": args.exp, "partner": args.partner, "scripts": args.scripts,
           "menu": menu, "runs": {}}
    for run in sorted(runs, key=lambda r: r.config["seed"]):
        seed = run.config["seed"]
        rows = []
        for step, f in checkpoints(run, prefix, args.n_ckpt):
            rel = osp.join(args.layout, "dishtraj", args.exp, str(seed))
            f.download(osp.join(pool, rel), replace=True)
            actor = osp.join(rel, f.name)
            ego = rg.Seat(("actor", config, actor), args.layout)
            if args.partner == "scripts":
                partners = [rg.Seat(("script", s), args.layout) for s in args.scripts]
            elif args.partner == "yml":
                partners = [rg.Seat(("actor", cfg, a), args.layout) for cfg, a in frozen]
            else:
                partners = [None]
            tally, pts, games = Counter(), 0.0, 0
            for pi, partner in enumerate(partners):
                for ep in range(args.episodes):
                    other = partner if stage2 else rg.Seat(("actor", config, actor), args.layout)
                    c, p = play(args.layout, ego, other, 1000 * seed + 37 * pi + ep, ep % 2)
                    tally.update(c)
                    pts += p
                    games += 1
            off = sum(v for k, v in tally.items() if k not in menu)
            rows.append({"step": step, "games": games, "per_game": {k: v / games for k, v in sorted(tally.items())},
                         "off_menu_share": off / max(1, sum(tally.values())), "points_per_game": pts / games})
            print(f"{args.exp} s{seed} step {step / 1e6:.2f}M: {games} games, points {pts / games:6.1f}, "
                  f"dishes/game {sum(tally.values()) / games:4.1f}, off-menu {rows[-1]['off_menu_share']:.2f}, "
                  + " ".join(f"{k}:{v / games:.1f}" for k, v in tally.most_common(4)), flush=True)
        out["runs"][str(seed)] = rows
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
