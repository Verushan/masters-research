"""How similar are the final agents, and do the agent types behave differently?

Two views, both from data already on disk:

1. Behaviour profiles, from the fill-in records. For every agent, and for each of
   the eight scripted partners from each seat, the profile holds: the tasks the
   agent completes per 100 steps (fill, dish, plate, deliver), how often it picks
   each of the six actions, and its action entropy. That is 8 x 2 x 11 = 176
   numbers per agent, z-scored across agents. From the profiles:
     - a 2-D PCA map of every agent, for the report's scatter plot;
     - mean profile distance within and between agent types;
     - for chosen pairs of types, an exact-or-sampled permutation test of
       "between-type distance exceeds within-type distance", with the training
       run as the unit;
     - leave-one-out nearest-neighbour accuracy: can an agent's type be told
       from its behaviour alone?
2. Compatibility, from the zero-shot cross-play matrices: how well agents of one
   type play with agents of another (mean return over distinct agent pairs, both
   seats). Agents that share conventions score well together; agents that
   merely look alike in a profile may not.

    python experiments/behaviour_similarity.py --layout unident_s \\
        --out experiments/results/behaviour_similarity_unident_s.json
"""

import argparse
import gzip
import itertools
import json
import os.path as osp
import re
from collections import defaultdict

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
RESULTS = osp.join(HERE, "results")

PARTNERS = ["potter", "server", "idle", "clutter", "generalist", "dial2", "dial5", "dial8"]
SEATS = [0, 1]
N_ACTIONS = 6
ACTION_NAMES = ["north", "south", "east", "west", "stay", "interact"]
TASK_NAMES = ["fill", "dish", "plate", "deliver"]

# (group key, label, fill-in folder, agent-name regex). Order is the report's.
GROUPS = [
    ("oracle", "Scripted oracle", "{layout}-step8", r"^ref_oracle_s\d+$"),
    ("s1_hand", "Stage-1 self-play, hand-shaped", "{layout}", r"^bench_sp_s\d+$"),
    ("s1_morl", "Stage-1 self-play, MORL", "{layout}", r"^bench_morl_ann-live3t_s\d+$"),
    ("s2_usual", "Usual stage-2 (FCP), hand-shaped", "{layout}", r"^s2_bench_sp_s\d+$"),
    ("s2_morl", "Usual stage-2 (FCP), MORL trainee", "{layout}", r"^s2_bench_sp-annego_s\d+$"),
    ("a", "A: specialists, with swaps", "{layout}-step8", r"^s2_scripted-hand_s\d+$"),
    ("ans", "A-ns: specialists, no swaps", "{layout}-step8", r"^s2_scripted-handns_s\d+$"),
    ("c2p", "C2′: corrected MORL, bonuses stay", "{layout}-step8", r"^s2_scripted-noann2_s\d+$"),
    ("c2", "C2: corrected MORL, bonuses fade", "{layout}-step8", r"^s2_scripted-neglect2_s\d+$"),
    ("cp", "C′: original MORL, bonuses stay", "{layout}-step8", r"^s2_scripted-noann_s\d+$"),
    ("c", "C: original MORL, bonuses fade", "{layout}-step8", r"^s2_scripted-neglect_s\d+$"),
    ("mix22", "Specialists + usual population, 22% of games", "{layout}-mixed", r"^s2_mixedspec-hand_s\d+$"),
    ("mix53", "Specialists + usual population, 53% of games", "{layout}-mixed", r"^s2_mixspec50-hand_s\d+$"),
]

# The comparisons the report makes. Each asks: are these two types' behaviours
# further apart than the spread within each type?
TESTS = [
    ("mix53", "a"),
    ("mix53", "s2_usual"),
    ("a", "s2_usual"),
    ("a", "ans"),
    ("a", "c2p"),
    ("a", "c"),
    ("c2p", "c"),
    ("s2_usual", "s2_morl"),
    ("s1_hand", "s1_morl"),
]

# Cross-play matrices whose agents include these types; later files add pairs
# the earlier ones lack, and duplicate cells are averaged.
CROSSPLAY = [
    "metrics_{layout}_hsp_scripted.json",
    "metrics_{layout}_hsp_scripted2.json",
    "metrics_{layout}_hsp_scripted_ns.json",
    "metrics_{layout}_s2_hsp_ego12.json",
]


def group_of(name):
    for key, _label, _folder, rx in GROUPS:
        if re.match(rx, name):
            return key
    return None


def profile(path):
    """The 176-number behaviour profile of one agent's fill-in record."""
    d = json.load(gzip.open(path))
    feats, names = [], []
    for partner in PARTNERS:
        for seat in SEATS:
            eps = [e for e in d["episodes"] if e["schedule"] == partner and e["seat"] == seat]
            if not eps:
                raise ValueError(f"{path}: no single-partner episodes vs {partner} from seat {seat}")
            steps = sum(len(e["action"]) for e in eps)
            done = np.zeros(4)
            acts = np.zeros(N_ACTIONS)
            ent = 0.0
            for e in eps:
                done += np.asarray(e["done"], dtype=float)[:, 0].sum(0)
                acts += np.bincount(np.asarray(e["action"]), minlength=N_ACTIONS)[:N_ACTIONS]
                ent += float(np.sum(e["entropy"]))
            feats += list(done / steps * 100) + list(acts / steps) + [ent / steps]
            names += [f"{partner}/s{seat}/{t}" for t in TASK_NAMES]
            names += [f"{partner}/s{seat}/act_{a}" for a in ACTION_NAMES]
            names += [f"{partner}/s{seat}/entropy"]
    return np.asarray(feats), names


def perm_test(D, idx_a, idx_b, n_samples=20000, seed=0):
    """Between-type minus within-type mean distance, and its permutation p-value.

    Labels are permuted across the pooled agents of the two types, keeping the
    group sizes. Exact when the number of splits is small enough, sampled
    otherwise; the observed split counts as one of the permutations either way.
    """
    pool = np.asarray(idx_a + idx_b)
    n_a = len(idx_a)

    def stat(mask):
        a, b = pool[mask], pool[~mask]
        between = D[np.ix_(a, b)].mean()
        within = np.concatenate([D[np.ix_(g, g)][np.triu_indices(len(g), 1)] for g in (a, b)]).mean()
        return between - within

    mask0 = np.zeros(len(pool), dtype=bool)
    mask0[:n_a] = True
    observed = stat(mask0)
    total = 1
    for i in range(n_a):
        total = total * (len(pool) - i) // (i + 1)
    if total <= 200000:
        stats = []
        for combo in itertools.combinations(range(len(pool)), n_a):
            m = np.zeros(len(pool), dtype=bool)
            m[list(combo)] = True
            stats.append(stat(m))
        stats = np.asarray(stats)
        exact = True
    else:
        rng = np.random.default_rng(seed)
        stats = [observed]
        for _ in range(n_samples):
            m = np.zeros(len(pool), dtype=bool)
            m[rng.choice(len(pool), n_a, replace=False)] = True
            stats.append(stat(m))
        stats = np.asarray(stats)
        exact = False
    p = float((stats >= observed - 1e-12).mean())
    return float(observed), p, exact


def compatibility(layout, agents_by_group):
    """Mean cross-play return between every pair of types, over distinct agents."""
    cells = defaultdict(list)
    for tmpl in CROSSPLAY:
        path = osp.join(RESULTS, tmpl.format(layout=layout))
        if not osp.exists(path):
            continue
        for key, v in json.load(open(path))["pair_matrix"].items():
            a, b = key.split("|")
            cells[(a, b)].append(v["mean"])
    cell = {k: float(np.mean(v)) for k, v in cells.items()}
    known = {a for pair in cell for a in pair}
    out = {}
    groups = [g for g in agents_by_group if any(a in known for a in agents_by_group[g])]
    for ga in groups:
        for gb in groups:
            # Both seat orders: a type pair's compatibility should not depend on
            # which of the two started in seat 0.
            vals = [
                cell[pair]
                for a in agents_by_group[ga]
                for b in agents_by_group[gb]
                if a != b
                for pair in ((a, b), (b, a))
                if pair in cell
            ]
            if vals:
                out[f"{ga}|{gb}"] = {"mean": float(np.mean(vals)), "n_pairs": len(vals)}
    self_play = {
        g: float(np.mean([cell[(a, a)] for a in agents_by_group[g] if (a, a) in cell]))
        for g in groups
        if any((a, a) in cell for a in agents_by_group[g])
    }
    return {"groups": groups, "cells": out, "self_play": self_play}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", default="unident_s")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    agents, groups, X, names = [], [], [], None
    for key, _label, folder, rx in GROUPS:
        folder_path = osp.join(RESULTS, "fillin", folder.format(layout=args.layout))
        if not osp.isdir(folder_path):
            continue
        for fn in sorted(f for f in __import__("os").listdir(folder_path) if f.endswith(".json.gz")):
            name = fn[: -len(".json.gz")]
            if not re.match(rx, name):
                continue
            feats, names = profile(osp.join(folder_path, fn))
            agents.append(name)
            groups.append(key)
            X.append(feats)
    X = np.asarray(X)
    print(f"{len(agents)} agents, {X.shape[1]} profile features")

    # z-score across agents; drop features no agent varies on.
    sd = X.std(0)
    keep = sd > 1e-9
    Z = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
    kept_names = [n for n, k in zip(names, keep) if k]
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).mean(-1))

    # PCA map.
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    coords = U[:, :2] * S[:2]
    var = S**2 / (S**2).sum()
    loadings = []
    for c in range(2):
        order = np.argsort(-np.abs(Vt[c]))[:8]
        loadings.append([{"feature": kept_names[i], "weight": float(Vt[c, i])} for i in order])

    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    order = [g for g, *_ in GROUPS if g in idx]

    dist = {}
    for ga in order:
        for gb in order:
            block = D[np.ix_(idx[ga], idx[gb])]
            if ga == gb:
                vals = block[np.triu_indices(len(idx[ga]), 1)]
            else:
                vals = block.ravel()
            dist[f"{ga}|{gb}"] = float(vals.mean()) if len(vals) else None

    tests = []
    for ga, gb in TESTS:
        if ga in idx and gb in idx:
            diff, p, exact = perm_test(D, idx[ga], idx[gb])
            tests.append(
                {
                    "a": ga,
                    "b": gb,
                    "within_a": dist[f"{ga}|{ga}"],
                    "within_b": dist[f"{gb}|{gb}"],
                    "between": dist[f"{ga}|{gb}"],
                    "between_minus_within": diff,
                    "p": p,
                    "exact": exact,
                }
            )

    # Leave-one-out nearest neighbour over the trained types (not the oracle).
    trained = [i for i, g in enumerate(groups) if g != "oracle"]
    hits = 0
    confusion = defaultdict(lambda: defaultdict(int))
    for i in trained:
        others = [j for j in trained if j != i]
        j = others[int(np.argmin(D[i, others]))]
        hits += groups[j] == groups[i]
        confusion[groups[i]][groups[j]] += 1
    sizes = np.array([len(idx[g]) for g in order if g != "oracle"])
    chance = float(((sizes / sizes.sum()) * ((sizes - 1) / (sizes.sum() - 1))).sum() / (sizes / sizes.sum()).sum())

    agents_by_group = {g: [agents[i] for i in idx[g]] for g in order}
    compat = compatibility(args.layout, agents_by_group)

    labels = {key: label for key, label, *_ in GROUPS}
    out = {
        "layout": args.layout,
        "groups": [{"key": g, "label": labels[g], "n": len(idx[g])} for g in order],
        "features": {"n_raw": int(X.shape[1]), "n_used": int(keep.sum()), "partners": PARTNERS},
        "agents": [
            {"agent": a, "group": g, "pc1": float(c[0]), "pc2": float(c[1])}
            for a, g, c in zip(agents, groups, coords)
        ],
        "pca": {"explained": [float(v) for v in var[:5]], "loadings": loadings},
        "distance": dist,
        "tests": tests,
        "nearest_neighbour": {
            "accuracy": hits / len(trained),
            "chance": chance,
            "n": len(trained),
            "confusion": {k: dict(v) for k, v in confusion.items()},
        },
        "compatibility": compat,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"PCA explained: {[round(v, 3) for v in var[:3]]}")
    print("\nwithin-type profile distance (lower = more alike)")
    for g in order:
        print(f"  {labels[g]:42s} n={len(idx[g]):2d}  within {dist[f'{g}|{g}']:.3f}")
    print("\nbetween vs within (permutation test, run as unit)")
    for t in tests:
        print(
            f"  {t['a']:>8s} vs {t['b']:<8s} between {t['between']:.3f}  within {t['within_a']:.3f}/{t['within_b']:.3f}"
            f"  diff {t['between_minus_within']:+.3f}  p={t['p']:.4f}{'' if t['exact'] else ' (sampled)'}"
        )
    print(f"\nnearest-neighbour type accuracy {hits}/{len(trained)} = {hits / len(trained):.2f} (chance {chance:.2f})")
    for g, row in confusion.items():
        print(f"  {g:>9s} -> {dict(row)}")
    print("\ncompatibility: mean cross-play return between types (distinct agents)")
    cg = compat["groups"]
    print("  " + " " * 10 + "".join(f"{g:>10s}" for g in cg))
    for ga in cg:
        row = "".join(
            f"{compat['cells'].get(f'{ga}|{gb}', {}).get('mean', float('nan')):10.1f}" for gb in cg
        )
        print(f"  {ga:>10s}{row}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
