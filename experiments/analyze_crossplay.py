#!/usr/bin/env python
"""Turn cross-play episode records into the benchmark's headline metrics.

`eval/cross_play.py` writes one row per episode; this collapses those rows into
the numbers the proposal's evaluation section asks for (4.2.7):

  self-play return        R(a, a) -- how good the agent is with a copy of itself
  ZSC return              mean over held-out partners, both seating orders
  worst-case return       the partner that goes worst, a robustness floor
  cross-partner spread    std of the per-partner means
  return stability        std across repeated rollouts of the *same* pair
  BR-Prox (proxy)         R(a, p) normalised by the best response to p that
                          exists anywhere in the evaluated pool

The BR-Prox here is a proxy and is labelled as one. True BR-Prox trains a fresh
best response per evaluation partner; this normalises instead by the strongest
partner-specific score achieved by any policy in the pool, which is a lower
bound on the true best response and therefore an *optimistic* proxy. It is still
the right shape of statistic -- it answers "how much of the achievable value with
this partner did the agent capture" rather than "how big is the raw number".

Returns are team returns: `eval_ep_sparse_r` already sums both players.

Usage:
    python experiments/analyze_crossplay.py \
        --records results/cross_play_deterministic.json results/cross_play_stochastic.json \
        --out results/metrics.json
"""

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OBJECTIVES = ["task_completion", "ingredient_prep", "plating", "coordination"]

# Display order and labels for the agent groups. The regex matches the policy
# names written by prep/gen_crossplay_yml.py.
GROUPS = [
    ("bench_sp", r"^bench_sp_s\d+$", "SP (sparse + hand-shaped)"),
    ("bench_sparse", r"^bench_sparse_s\d+$", "SP (sparse only)"),
    ("bench_morl", r"^bench_morl_s\d+$", "MORL (fixed w, final)"),
    ("bench_morl_peak", r"^bench_morl_s\d+_peak$", "MORL (fixed w, peak)"),
    ("bench_morl_ad", r"^bench_morl_ad_s\d+$", "MORL (adaptive w, final)"),
    ("bench_morl_ad_peak", r"^bench_morl_ad_s\d+_peak$", "MORL (adaptive w, peak)"),
    ("s2_bench_sp", r"^s2_bench_sp_s\d+$", "FCP S2 on SP population"),
    ("s2_bench_morl", r"^s2_bench_morl_s\d+$", "FCP S2 on MORL (fixed w) population"),
    ("s2_bench_morl_ad", r"^s2_bench_morl_ad_s\d+$", "FCP S2 on MORL (adaptive w) population"),
    ("s2_mixed", r"^s2_mixed_s\d+$", "FCP S2 on mixed population"),
    # pid-ladder.sh rungs. The arm name carries the rung ("bench_sp-rung1"), so
    # the pool entry is `s2_bench_sp-rung1_s{seed}` and the plain s2_bench_sp
    # pattern above -- anchored on `_s\d+$` -- does not match it.
    (
        "s2_pid_rung1",
        r"^s2_bench_sp-rung1_s\d+$",
        "FCP S2 - partner id: critic only (control)",
    ),
    (
        "s2_pid_rung2",
        r"^s2_bench_sp-rung2_s\d+$",
        "FCP S2 - partner id: actor sees raw scalar",
    ),
    (
        "s2_pid_rung3",
        r"^s2_bench_sp-rung3_s\d+$",
        "FCP S2 - partner id: actor sees one-hot",
    ),
    ("fcp_s2", r"^fcp_s2_s\d+$", "FCP stage-2 (pipeline, 16x3 SP)"),
    ("heldout", r"^heldout_sp", "Held-out partners (self-play)"),
    ("heldout_hsp", r"^heldout_hsp", "Held-out partners (HSP bias agents)"),
]


def group_of(name):
    for key, pattern, _ in GROUPS:
        if re.match(pattern, name):
            return key
    return "other"


def load(paths):
    rows = []
    for path in paths:
        # The raw episode dumps compress ~35x and are committed gzipped, so accept
        # either form and let the caller name whichever is on disk.
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            blob = json.load(f)
        tag = "stochastic" if blob.get("eval_stochastic") else "deterministic"
        for record in blob["records"]:
            record["mode"] = tag
            rows.append(record)
    return rows


def cell_stats(rows):
    """(agent0, agent1) -> per-cell aggregates, keeping both seating orders apart."""
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[(r["agent0"], r["agent1"])].append(r)

    cells = {}
    for pair, records in by_pair.items():
        returns = np.array([r["eval_ep_sparse_r"] for r in records], dtype=float)
        cell = {
            "n": len(records),
            "mean": float(returns.mean()),
            "std": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
            "min": float(returns.min()),
            "max": float(returns.max()),
        }
        for objective in OBJECTIVES:
            key = f"eval_ep_obj_{objective}"
            if key in records[0]:
                cell[objective] = float(np.mean([r[key] for r in records]))
        cells[pair] = cell
    return cells


def symmetrised(cells, ego, partner):
    """Mean return for ego with partner, averaging over which side ego sits on.

    random0 is not position-symmetric, so a policy can look strong from one seat
    and useless from the other; a single order would report whichever half the
    pair list happened to contain.
    """
    values = [
        cells[(a, b)]["mean"]
        for a, b in [(ego, partner), (partner, ego)]
        if (a, b) in cells
    ]
    return float(np.mean(values)) if values else None


def objective_profile(cells, ego, partners):
    """Mean per-objective team totals for ego across the given partners."""
    profile = {}
    for objective in OBJECTIVES:
        values = []
        for partner in partners:
            for pair in [(ego, partner), (partner, ego)]:
                if pair in cells and objective in cells[pair]:
                    values.append(cells[pair][objective])
        if values:
            profile[objective] = float(np.mean(values))
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--partner_group",
        default="heldout",
        help="Group used as the zero-shot evaluation partner set.",
    )
    args = parser.parse_args()

    rows = load(args.records)
    agents = sorted({r["agent0"] for r in rows} | {r["agent1"] for r in rows})
    members = defaultdict(list)
    for agent in agents:
        members[group_of(agent)].append(agent)

    all_cells = cell_stats(rows)
    det_cells = cell_stats([r for r in rows if r["mode"] == "deterministic"])
    sto_rows = [r for r in rows if r["mode"] == "stochastic"]
    sto_cells = cell_stats(sto_rows)

    partners = sorted(members.get(args.partner_group, []))
    assert partners, f"no agents in partner group {args.partner_group!r}"

    # Best response available anywhere in the pool for each partner -- the
    # denominator of the BR-Prox proxy. Held-out partners are allowed to be their
    # own best response, which is usually what they are.
    br_hat = {}
    for partner in partners:
        scores = [
            symmetrised(det_cells, other, partner)
            for other in agents
            if other != partner
        ]
        scores = [s for s in scores if s is not None]
        # A partner paired with itself is the self-play ceiling and the strongest
        # response that exists, so it belongs in the max.
        self_play = det_cells.get((partner, partner), {}).get("mean")
        if self_play is not None:
            scores.append(self_play)
        br_hat[partner] = max(scores) if scores else 0.0

    per_agent = {}
    for agent in agents:
        vs_partners = {}
        for partner in partners:
            if partner == agent:
                continue
            value = symmetrised(det_cells, agent, partner)
            if value is not None:
                vs_partners[partner] = value
        values = np.array(list(vs_partners.values()), dtype=float)

        stability = [
            sto_cells[pair]["std"]
            for pair in sto_cells
            if agent in pair and sto_cells[pair]["n"] > 1
        ]
        br_prox = [
            vs_partners[p] / br_hat[p] for p in vs_partners if br_hat.get(p, 0) > 0
        ]

        per_agent[agent] = {
            "group": group_of(agent),
            "self_play": det_cells.get((agent, agent), {}).get("mean"),
            "self_play_stochastic": sto_cells.get((agent, agent), {}).get("mean"),
            "zsc_mean": float(values.mean()) if values.size else None,
            "zsc_worst": float(values.min()) if values.size else None,
            "zsc_best": float(values.max()) if values.size else None,
            "zsc_spread": float(values.std(ddof=1)) if values.size > 1 else None,
            "return_stability": float(np.mean(stability)) if stability else None,
            "br_prox_proxy": float(np.mean(br_prox)) if br_prox else None,
            "vs_partners": vs_partners,
            "objectives_self_play": objective_profile(det_cells, agent, [agent]),
            "objectives_vs_partners": objective_profile(det_cells, agent, partners),
        }

    def agg(group, field):
        values = [
            per_agent[a][field] for a in members.get(group, []) if per_agent[a][field] is not None
        ]
        if not values:
            return None
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n": len(values),
        }

    per_group = {}
    for key, _, label in GROUPS:
        if not members.get(key):
            continue
        per_group[key] = {
            "label": label,
            "members": sorted(members[key]),
            **{
                field: agg(key, field)
                for field in [
                    "self_play",
                    "self_play_stochastic",
                    "zsc_mean",
                    "zsc_worst",
                    "zsc_spread",
                    "return_stability",
                    "br_prox_proxy",
                ]
            },
        }
        for name, source in [
            ("objectives_self_play", "objectives_self_play"),
            ("objectives_vs_partners", "objectives_vs_partners"),
        ]:
            profile = {}
            for objective in OBJECTIVES:
                values = [
                    per_agent[a][source][objective]
                    for a in members[key]
                    if objective in per_agent[a][source]
                ]
                if values:
                    profile[objective] = float(np.mean(values))
            per_group[key][name] = profile

    # Group x group mean return, both seating orders folded together.
    group_matrix = {}
    for row_key, _, _ in GROUPS:
        if not members.get(row_key):
            continue
        group_matrix[row_key] = {}
        for col_key, _, _ in GROUPS:
            if not members.get(col_key):
                continue
            values = []
            for a in members[row_key]:
                for b in members[col_key]:
                    if a == b:
                        continue
                    value = symmetrised(det_cells, a, b)
                    if value is not None:
                        values.append(value)
            group_matrix[row_key][col_key] = float(np.mean(values)) if values else None

    result = {
        "agents": agents,
        "groups": {k: sorted(v) for k, v in members.items()},
        "group_labels": {k: label for k, _, label in GROUPS},
        "partner_group": args.partner_group,
        "partners": partners,
        "br_hat": br_hat,
        "per_agent": per_agent,
        "per_group": per_group,
        "group_matrix": group_matrix,
        "pair_matrix": {f"{a}|{b}": v for (a, b), v in det_cells.items()},
        "pair_matrix_stochastic": {f"{a}|{b}": v for (a, b), v in sto_cells.items()},
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)

    # Console summary
    def fmt(v, nd=1):
        return "  -  " if v is None else f"{v:.{nd}f}"

    print(f"\n{'group':32} {'self-play':>10} {'ZSC mean':>10} {'ZSC worst':>10} {'spread':>8} {'BR-Prox*':>9}")
    print("-" * 84)
    for key in per_group:
        g = per_group[key]
        print(
            f"{g['label']:32} "
            f"{fmt((g['self_play'] or {}).get('mean')):>10} "
            f"{fmt((g['zsc_mean'] or {}).get('mean')):>10} "
            f"{fmt((g['zsc_worst'] or {}).get('mean')):>10} "
            f"{fmt((g['zsc_spread'] or {}).get('mean')):>8} "
            f"{fmt((g['br_prox_proxy'] or {}).get('mean'), 3):>9}"
        )

    print(f"\nself-play objective profile (team totals per episode)")
    print(f"{'group':32} " + " ".join(f"{o[:12]:>13}" for o in OBJECTIVES))
    print("-" * 84)
    for key in per_group:
        g = per_group[key]
        print(f"{g['label']:32} " + " ".join(f"{fmt(g['objectives_self_play'].get(o)):>13}" for o in OBJECTIVES))

    print(f"\ngroup x group mean return (deterministic, both orders)")
    keys = list(group_matrix)
    print(f"{'':22}" + "".join(f"{k[:14]:>15}" for k in keys))
    for row in keys:
        print(f"{row[:22]:22}" + "".join(f"{fmt(group_matrix[row][c]):>15}" for c in keys))

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
