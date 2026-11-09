"""Did a training run converge, and did it get worse afterwards? (pre-registered 10 Nov 2026)

Rules: experiments/report/preregistration-2026-11-10-convergence.md. In short, for every
finished run of an experiment:

- replay ~25 evenly spaced saved checkpoints offline, sampled actions, on two independent
  game sets A and B whose seeds are identical for every checkpoint (common random numbers);
- choose the peak on set A (3-checkpoint running mean), measure it on set B, so the peak is
  never both chosen and scored on the same games;
- classify the run: not converged / stable / declined / collapsed, with the collapse time;
- record a state bank from the peak checkpoint and measure behavioural drift: mean
  KL(pi_peak || pi_c) on those states for every checkpoint c, and KL between consecutive
  checkpoints (sudden break vs gradual drift);
- apply the same rules to the W&B training curve, so the cheap measure can be validated.

Partners as in dish_trajectory.py: stage 1 plays itself; stage 2 plays each training partner
(scripted specialists, or every --yml_every-th frozen entry of the population yml) once per seat.

    python experiments/convergence_audit.py --layout pantry_cx --exp bench_sp-cx-lr --partner self \\
        --out experiments/results/convergence/pantry_cx_bench_sp-cx-lr.json
    python experiments/convergence_audit.py --layout pantry_cx --exp fcp-S2-scripted-hand-v2 --partner scripts \\
        --scripts potter server idle dial2 dial8 onion_potter tomato_potter --out ...
"""

import argparse
import json
import os
import os.path as osp
import random
import re
import sys
from multiprocessing import get_context

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, osp.join(HERE, "..", "zsc-eval"))

COMPETENT = 50.0      # same bar as the held-out partner filter
DECLINE = 0.75        # final <= 0.75 x confirmed peak
ALPHA = 0.05
SET_SEEDS = {"A": 7000, "B": 9000}


# ---------------------------------------------------------------- classification


def running_mean(x, k=3):
    x = np.asarray(x, dtype=float)
    return np.array([x[max(0, i - k // 2): i + k // 2 + 1].mean() for i in range(len(x))])


def welch_one_sided(peak_games, final_games):
    """p for H1: mean(final) < mean(peak)."""
    from scipy import stats

    a, b = np.asarray(peak_games, float), np.asarray(final_games, float)
    if a.std() == 0 and b.std() == 0:
        return 0.0 if b.mean() < a.mean() else 1.0
    t, p = stats.ttest_ind(b, a, equal_var=False)
    return float(p / 2 if t < 0 else 1 - p / 2)


def classify(steps, select, confirm, confirm_games=None):
    """Pre-registered rule. `select` picks the peak, `confirm` measures it.

    Offline: select = R_A, confirm = R_B, confirm_games = per-checkpoint set-B returns.
    Training curve: select = confirm = the logged curve, no games (no significance test).
    """
    steps, select, confirm = list(steps), np.asarray(select, float), np.asarray(confirm, float)
    c = int(np.argmax(running_mean(select)))
    peak = float(confirm[c])
    last = list(range(max(0, len(steps) - 3), len(steps)))
    final = float(confirm[last].mean())
    out = {"peak_step": steps[c], "peak": peak, "final": final, "drop": (peak - final) / peak if peak > 0 else 0.0}
    if peak < COMPETENT:
        out["class"] = "not converged"
        return out
    p = None
    if confirm_games is not None and c not in last:
        p = welch_one_sided(confirm_games[c], [g for i in last for g in confirm_games[i]])
    out["p_decline"] = p
    declined = final <= DECLINE * peak and (p is None or p < ALPHA) and c not in last
    out["class"] = ("collapsed" if final < COMPETENT else "declined") if declined else "stable"
    rm = running_mean(confirm)
    below = rm < DECLINE * peak
    for i in range(c + 1, len(steps)):
        if below[i:].all():
            out["collapse_step"] = steps[i]
            break
    return out


# ---------------------------------------------------------------- playing


def _seat(rg, spec, layout):
    return rg.Seat(spec, layout)


def probs_of(seat, obs, avail):
    import torch

    with torch.no_grad():
        probs, rnn = seat.actor.get_probs(np.asarray(obs, dtype=np.float32)[None], seat.rnn,
                                          np.ones((1, 1), dtype=np.float32), np.asarray(avail)[None])
    seat.rnn = rnn.cpu().numpy()
    p = probs.cpu().numpy().reshape(-1).astype(np.float64)
    return p / p.sum()


def play(rg, env, layout, seats, seed, record_seats=()):
    """One 400-step game. Returns (sparse return, dishes, {seat: [(obs, avail)]})."""
    import torch

    from dish_trajectory import DELIVERED, code

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs, _share, avail = env.reset()
    scripts = [None, None]
    for i, s in enumerate(seats):
        s.reset()
        if s.kind == "script":
            scripts[i] = rg.make_script(s.name, layout)
            scripts[i].reset(env.base_mdp, env.base_env.state, i)
    env.script_agent = scripts
    DELIVERED.clear()
    trace = {i: [] for i in record_seats}
    ret = 0.0
    for _t in range(400):
        for i in record_seats:
            trace[i].append((np.array(obs[i]), np.array(avail[i])))
        joint = [[s.act(obs[i], avail[i], False, rng)] if s.kind == "actor" else [4] for i, s in enumerate(seats)]
        obs, _share, _r, dones, info, avail = env.step(np.array(joint))
        ret += float(sum(info["sparse_r_by_agent"]))
        if np.all(dones):
            break
    dishes = {}
    for d in DELIVERED:
        dishes[code(d)] = dishes.get(code(d), 0) + 1
    return ret, dishes, trace


def games(partners, n_self):
    """(partner index, ego seat) per game in a set: n_self self-play games, or each partner x seat."""
    if partners is None:
        return [(None, 0)] * n_self
    return [(pi, seat) for pi in range(len(partners)) for seat in (0, 1)]


def audit_run(job):
    """Worker: every checkpoint of one run on both sets, then the drift measurement."""
    import torch

    torch.set_num_threads(1)
    import render_gameplay as rg

    layout, config, ckpts, partner_specs, n_self, n_bank = (job[k] for k in
                                                           ("layout", "config", "ckpts", "partners", "n_self", "n_bank"))
    env = rg.make_env(layout)
    partners = None if partner_specs is None else [_seat(rg, tuple(p), layout) for p in partner_specs]
    plan = games(partners, n_self)
    rows = []
    for step, actor in ckpts:
        ego = _seat(rg, ("actor", config, actor), layout)
        row = {"step": step}
        for name, base in SET_SEEDS.items():
            rets, dishes = [], {}
            for g, (pi, seat) in enumerate(plan):
                other = _seat(rg, ("actor", config, actor), layout) if pi is None else partners[pi]
                pair = [ego, other] if seat == 0 else [other, ego]
                r, d, _ = play(rg, env, layout, pair, base + g)
                rets.append(r)
                for k, v in d.items():
                    dishes[k] = dishes.get(k, 0) + v
            row[name] = rets
            row[f"dishes_{name}"] = dishes
        rows.append(row)
    steps = [r["step"] for r in rows]
    off = classify(steps, [np.mean(r["A"]) for r in rows], [np.mean(r["B"]) for r in rows], [r["B"] for r in rows])

    # Drift: a state bank from the peak checkpoint's own games, replayed through every checkpoint.
    peak_actor = dict(ckpts)[off["peak_step"]]
    peak = _seat(rg, ("actor", config, peak_actor), layout)
    bank = []
    for g in range(n_bank):
        pi, seat = plan[g % len(plan)]
        other = _seat(rg, ("actor", config, peak_actor), layout) if pi is None else partners[pi]
        pair = [peak, other] if seat == 0 else [other, peak]
        _r, _d, trace = play(rg, env, layout, pair, 5000 + g, record_seats=(seat,))
        bank.append(trace[seat])

    def dist(actor):
        s = _seat(rg, ("actor", config, actor), layout)
        out = []
        for seq in bank:
            s.reset()
            out.append(np.array([probs_of(s, o, a) for o, a in seq]))
        return out

    def kl(p, q):
        p, q = np.clip(p, 1e-8, 1), np.clip(q, 1e-8, 1)
        return float(np.mean(np.concatenate([(pp * np.log(pp / qq)).sum(-1) for pp, qq in zip(p, q)])))

    ref = dist(peak_actor)
    dists = [dist(a) for _s, a in ckpts]
    for i, row in enumerate(rows):
        row["kl_from_peak"] = kl(ref, dists[i])
        row["kl_from_prev"] = kl(dists[i - 1], dists[i]) if i else 0.0
        row["entropy_on_bank"] = float(np.mean(np.concatenate([-(np.clip(d, 1e-8, 1) * np.log(np.clip(d, 1e-8, 1))).sum(-1)
                                                               for d in dists[i]])))
    ci = steps.index(off["peak_step"])
    after = [r["kl_from_prev"] for r in rows[ci + 1:]]
    total = rows[-1]["kl_from_peak"]
    off["drift_total_kl"] = total
    off["drift_max_step_kl"] = max(after) if after else 0.0
    off["drift"] = "sudden" if after and max(after) > 0.5 * total else "gradual"
    return {"seed": job["seed"], "checkpoints": rows, "offline": off}


# ---------------------------------------------------------------- main


def training_curve(run, stage2):
    h = run.history(samples=100000, pandas=True)
    if stage2:
        cols = [c for c in h.columns if re.fullmatch(r"either-.*-ep_sparse_r", c)]
    else:
        cols = ["ep_sparse_r"] if "ep_sparse_r" in h.columns else []
    if not cols:
        return [], []
    v = h[cols].mean(axis=1)
    keep = v.notna()
    return [int(s) for s in h["_step"][keep]], [float(x) for x in v[keep]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--partner", choices=["self", "scripts", "yml"], required=True)
    ap.add_argument("--scripts", nargs="*", default=[])
    ap.add_argument("--yml", default=None)
    ap.add_argument("--yml_every", type=int, default=3)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--n_ckpt", type=int, default=25)
    ap.add_argument("--n_self", type=int, default=8, help="Self-play games per set (stage 1)")
    ap.add_argument("--n_bank", type=int, default=4, help="Games of the peak checkpoint in the state bank")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--project", default="Overcooked-new")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import wandb

    import render_gameplay as rg
    from dish_trajectory import checkpoints

    pool = os.environ["POLICY_POOL"]
    api = wandb.Api(timeout=120)
    runs = [r for r in api.runs(f"{os.environ['WANDB_ENTITY']}/{args.project}",
                                filters={"config.experiment_name": args.exp, "config.layout_name": args.layout,
                                         "state": "finished"})
            if args.seeds is None or r.config["seed"] in args.seeds]
    stage2 = args.partner != "self"
    prefix, config = ("fcp_adaptive/", rg.RNN) if stage2 else ("", rg.MLP)
    if args.partner == "scripts":
        partners = [("script", s) for s in args.scripts]
    elif args.partner == "yml":
        import yaml

        pop = yaml.safe_load(open(osp.join(pool, args.yml)))
        partners = [("actor", v["policy_config_path"], v["model_path"]["actor"]) for v in pop.values()
                    if not v.get("train") and "model_path" in v][:: args.yml_every]
    else:
        partners = None

    jobs, curves = [], {}
    seeds_seen = set()
    for run in sorted(runs, key=lambda r: r.config["seed"]):
        seed = run.config["seed"]
        if seed in seeds_seen:
            raise SystemExit(f"two finished runs with seed {seed}; tag the unwanted one 'unused'")
        seeds_seen.add(seed)
        ckpts = []
        for step, f in checkpoints(run, prefix, args.n_ckpt):
            rel = osp.join(args.layout, "convaudit", args.exp, str(seed))
            if not osp.exists(osp.join(pool, rel, f.name)):
                f.download(osp.join(pool, rel), replace=True)
            ckpts.append((step, osp.join(rel, f.name)))
        curves[seed] = training_curve(run, stage2)
        jobs.append({"seed": seed, "layout": args.layout, "config": config, "ckpts": ckpts, "partners": partners,
                     "n_self": args.n_self, "n_bank": args.n_bank})
        print(f"{args.exp} s{seed}: {len(ckpts)} checkpoints", flush=True)

    out = {"layout": args.layout, "exp": args.exp, "partner": args.partner, "partners": partners,
           "rules": {"competent": COMPETENT, "decline": DECLINE, "alpha": ALPHA, "set_seeds": SET_SEEDS},
           "runs": {}}
    with get_context("spawn").Pool(min(args.jobs, len(jobs))) as p:
        for res in p.imap_unordered(audit_run, jobs):
            seed = res["seed"]
            steps, vals = curves[seed]
            res["training_curve"] = {"step": steps, "ep_sparse_r": vals}
            res["from_curve"] = classify(steps, vals, vals) if vals else None
            out["runs"][str(seed)] = res
            o = res["offline"]
            print(f"{args.exp} s{seed}: {o['class']:13s} peak {o['peak']:6.1f} @ {o['peak_step'] / 1e6:.2f}M, "
                  f"final {o['final']:6.1f}, drift {o['drift']} (KL {o['drift_total_kl']:.2f}); "
                  f"training curve says {res['from_curve']['class'] if res['from_curve'] else '-'}", flush=True)
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
