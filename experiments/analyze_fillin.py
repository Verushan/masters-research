"""Step 4 of the Fill-In Evaluation Suite: turn fill-in episodes into metrics.

Reads the records `experiments/fillin_eval.py` writes (one file per agent) and
reports, per agent and per arm:

soups/100       Team deliveries per 100 steps, the outcome everything else
                explains. Per partner, and per segment for swap schedules.
coverage        Of the tasks a specialist partner neglects, the fraction the
                agent performs at least once per 100 steps. 1.0 = the agent
                picks up everything left undone; 0 = none of it. Defined only
                against partners who neglect something (potter, server, idle,
                clutter).
duplication     Share of the agent's completed tasks that its partner already
                covers. An agent filling pots beside a potter scores 1.0.
latency         Steps after a swap until the agent first does a task the new
                partner neglects. Censored at the segment's end (reported as
                the segment length) when it never does.
seat share      How much of the variation in the agent's pot-filling share is
                explained by which seat it started in, versus which partner it
                has (eta-squared, over preference partners and both seats).
                Near 1 = its role is set by where it starts, not by who it is
                playing with.
entropy         Mean entropy of the agent's action distribution (nats; the
                maximum over six actions is 1.79), per partner.
stay / blocked  Fraction of steps the agent stood still / walked into its partner.

Arms are compared on seed-level values with the exact permutation test from
compare_arms.py, the training run being the unit of replication.

    python experiments/analyze_fillin.py experiments/results/fillin/unident_s_*.json.gz \\
        --out experiments/results/fillin_metrics_unident_s.json
"""

import argparse
import gzip
import json
import os.path as osp
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from compare_arms import permutation_test  # noqa: E402

NEGLECT_PARTNERS = ("potter", "server", "idle", "clutter")
PREFERENCE_PARTNERS = ("generalist", "dial2", "dial5", "dial8")
RATE_THRESHOLD = 1.0  # completions per 100 steps that count as "does this task"


def arm_of(agent_name):
    """'bench_morl_fill-live3t_s3' -> 'bench_morl_fill-live3t'."""
    return re.sub(r"_s\d+(?:r\d+)?$", "", agent_name)


def segments(ep):
    """Contiguous runs of one partner: (partner, start, end, index-in-episode)."""
    partners = ep["partner"]
    out, start = [], 0
    for t in range(1, len(partners) + 1):
        if t == len(partners) or partners[t] != partners[start]:
            out.append((partners[start], start, t, len(out)))
            start = t
    return out


def segment_metrics(ep, partner, start, end, idx, tasks, covers):
    done = np.asarray(ep["done"][start:end])  # (T, 2, 4), agent first
    demand = np.asarray(ep["demand"][start:end])  # (T, 4)
    steps = end - start
    agent, part = done[:, 0].sum(0), done[:, 1].sum(0)
    neglected = [i for i, t in enumerate(tasks) if t not in covers]
    covered = [i for i, t in enumerate(tasks) if t in covers]
    rate = agent / steps * 100

    m = {
        "partner": partner,
        "segment": idx,
        "steps": steps,
        "soups_per_100": float(done[:, :, 3].sum() / steps * 100),
        "agent_tasks": agent.tolist(),
        "partner_tasks": part.tolist(),
        "demand_present": (demand > 0).mean(0).tolist(),
        "entropy": float(np.mean(ep["entropy"][start:end])),
        "stay": float(np.mean(ep["stay"][start:end])),
        "blocked": float(np.mean(ep["blocked"][start:end])),
        "coverage": float(np.mean(rate[neglected] >= RATE_THRESHOLD)) if neglected and partner in NEGLECT_PARTNERS else None,
        "duplication": float(agent[covered].sum() / agent.sum()) if agent.sum() > 0 and covered else None,
        "pot_share": float(agent[0] / agent.sum()) if agent.sum() > 0 else None,
    }
    # Latency only means something after a swap, and only toward a partner
    # who leaves something undone.
    if idx > 0 and neglected:
        hits = np.nonzero(done[:, 0][:, neglected].sum(1) > 0)[0]
        m["latency"] = int(hits[0]) if len(hits) else steps
        m["latency_censored"] = not len(hits)
    return m


def agent_summary(payload):
    tasks = payload["tasks"]
    covers = {k: v["covers"] for k, v in payload["partners"].items()}
    segs = []
    for ep in payload["episodes"]:
        for partner, start, end, idx in segments(ep):
            m = segment_metrics(ep, partner, start, end, idx, tasks, covers[partner])
            m.update(seat=ep["seat"], schedule=ep["schedule"])
            segs.append(m)

    def mean(key, pred):
        vals = [s[key] for s in segs if pred(s) and s.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    whole = lambda s: s["schedule"] == s["partner"]  # noqa: E731  single-partner episodes
    swapped = lambda s: s["segment"] > 0  # noqa: E731

    out = {
        "agent": payload["agent"],
        "arm": arm_of(payload["agent"]),
        "soups_vs_neglect": mean("soups_per_100", lambda s: whole(s) and s["partner"] in NEGLECT_PARTNERS),
        "soups_vs_preference": mean("soups_per_100", lambda s: whole(s) and s["partner"] in PREFERENCE_PARTNERS),
        "soups_after_swap": mean("soups_per_100", swapped),
        "coverage": mean("coverage", lambda s: whole(s)),
        "duplication": mean("duplication", lambda s: whole(s) and s["partner"] in ("potter", "server")),
        "latency": mean("latency", swapped),
        "latency_censored": mean("latency_censored", swapped),
        "entropy": mean("entropy", lambda s: True),
        "stay": mean("stay", lambda s: True),
        "blocked": mean("blocked", lambda s: True),
        "by_partner": {},
    }
    for p in NEGLECT_PARTNERS + PREFERENCE_PARTNERS:
        sel = lambda s, p=p: whole(s) and s["partner"] == p  # noqa: E731
        out["by_partner"][p] = {
            k: mean(k, sel)
            for k in ("soups_per_100", "coverage", "duplication", "pot_share", "entropy", "stay", "blocked")
        }
        for seat in (0, 1):
            here = lambda s, p=p, seat=seat: whole(s) and s["partner"] == p and s["seat"] == seat  # noqa: E731
            out["by_partner"][p][f"pot_share_seat{seat}"] = mean("pot_share", here)
            out["by_partner"][p][f"soups_seat{seat}"] = mean("soups_per_100", here)
        # Where the game stalls: the fraction of steps each task sat needed.
        # A soup-plating value near 1 means finished soups waited in the pot
        # most of the episode because nobody came to plate them.
        dem = [s["demand_present"] for s in segs if sel(s)]
        out["by_partner"][p]["waiting"] = np.mean(dem, 0).tolist() if dem else None

    # Seat vs partner: two-way decomposition of the agent's pot-filling share
    # over the preference partners (who leave the agent free to choose) and
    # both seats. Cells the agent did nothing in are skipped.
    cells = defaultdict(list)
    for s in segs:
        if whole(s) and s["partner"] in PREFERENCE_PARTNERS and s["pot_share"] is not None:
            cells[(s["seat"], s["partner"])].append(s["pot_share"])
    grid = {k: np.mean(v) for k, v in cells.items()}
    seats = sorted({k[0] for k in grid})
    partners = sorted({k[1] for k in grid})
    if len(seats) == 2 and len(partners) >= 2 and all((a, b) in grid for a in seats for b in partners):
        y = np.array([[grid[(a, b)] for b in partners] for a in seats])
        grand = y.mean()
        ss_total = ((y - grand) ** 2).sum()
        ss_seat = y.shape[1] * ((y.mean(1) - grand) ** 2).sum()
        ss_partner = y.shape[0] * ((y.mean(0) - grand) ** 2).sum()
        out["seat_share"] = float(ss_seat / ss_total) if ss_total > 0 else None
        out["partner_share"] = float(ss_partner / ss_total) if ss_total > 0 else None
    else:
        out["seat_share"] = out["partner_share"] = None
    return out


KEYS = [
    ("soups_vs_neglect", "soups/100 vs neglecters"),
    ("soups_vs_preference", "soups/100 vs preference"),
    ("soups_after_swap", "soups/100 after a swap"),
    ("coverage", "coverage of neglected"),
    ("duplication", "duplication"),
    ("latency", "switch latency (steps)"),
    ("seat_share", "role set by seat"),
    ("entropy", "action entropy"),
    ("blocked", "blocked"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--base", default="bench_sp", help="Arm every other is compared against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    agents = [agent_summary(json.load(gzip.open(f, "rt"))) for f in args.files]
    by_arm = defaultdict(list)
    for a in agents:
        by_arm[a["arm"]].append(a)

    arms = {}
    for arm, members in sorted(by_arm.items()):
        arms[arm] = {"n": len(members)}
        for key, _ in KEYS:
            vals = [m[key] for m in members if m[key] is not None]
            arms[arm][key] = {
                "mean": float(np.mean(vals)) if vals else None,
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "values": vals,
            }
        def pooled(p, k):
            vals = [m["by_partner"][p][k] for m in members if m["by_partner"][p][k] is not None]
            if not vals:
                return None
            # Vectors (per-task waiting) average elementwise; scalars to a scalar.
            return np.mean(vals, 0).tolist() if isinstance(vals[0], list) else float(np.mean(vals))

        arms[arm]["by_partner"] = {
            p: {k: pooled(p, k) for k in members[0]["by_partner"][p]} for p in members[0]["by_partner"]
        }

    width = max(len(a) for a in arms) + 2
    print(f"\n{'arm':{width}s} n  " + "  ".join(f"{label[:14]:>14s}" for _, label in KEYS))
    for arm, d in arms.items():
        cells = []
        for key, _ in KEYS:
            v = d[key]["mean"]
            cells.append(f"{v:14.2f}" if v is not None else f"{'-':>14s}")
        print(f"{arm:{width}s} {d['n']:<2d} " + "  ".join(cells))

    print("\nsoups/100 by partner (single-partner episodes, both seats)")
    ps = NEGLECT_PARTNERS + PREFERENCE_PARTNERS
    print(f"{'arm':{width}s} " + "".join(f"{p:>11s}" for p in ps))
    for arm, d in arms.items():
        print(
            f"{arm:{width}s} "
            + "".join(
                f"{d['by_partner'][p]['soups_per_100']:11.2f}" if d["by_partner"][p]["soups_per_100"] is not None else f"{'-':>11s}"
                for p in ps
            )
        )

    print("\nsoups/100 by seat (seat 0 / seat 1), neglect partners")
    for arm, d in arms.items():
        pairs = []
        for p in NEGLECT_PARTNERS:
            a, b = d["by_partner"][p]["soups_seat0"], d["by_partner"][p]["soups_seat1"]
            pairs.append(f"{p}:{a:.2f}/{b:.2f}" if a is not None and b is not None else f"{p}:-")
        print(f"{arm:{width}s} " + "  ".join(pairs))

    print("\npot-filling share by seat (seat 0 / seat 1), preference partners")
    for arm, d in arms.items():
        pairs = []
        for p in PREFERENCE_PARTNERS:
            a, b = d["by_partner"][p]["pot_share_seat0"], d["by_partner"][p]["pot_share_seat1"]
            pairs.append(f"{p}:{a:.2f}/{b:.2f}" if a is not None and b is not None else f"{p}:-")
        print(f"{arm:{width}s} " + "  ".join(pairs))

    print("\nwhere games stall: fraction of steps each task sat needed (fill / dish / plate / deliver)")
    for arm, d in arms.items():
        parts = []
        for p in ("potter", "server", "idle"):
            w = d["by_partner"][p]["waiting"]
            parts.append(f"{p}:" + "/".join(f"{x:.2f}" for x in w) if w else f"{p}:-")
        print(f"{arm:{width}s} " + "  ".join(parts))

    comparisons = {}
    if args.base in arms:
        print(f"\nagainst {args.base} (exact permutation test on seed-level values)")
        for arm in arms:
            if arm == args.base:
                continue
            comparisons[arm] = {}
            for key in ("soups_vs_neglect", "soups_after_swap", "coverage", "seat_share"):
                a, b = arms[arm][key]["values"], arms[args.base][key]["values"]
                if len(a) < 2 or len(b) < 2:
                    continue
                diff, p, _exact, _n, _floor = permutation_test(a, b)
                comparisons[arm][key] = {"diff": float(diff), "p": float(p)}
            line = "  ".join(f"{k} {v['diff']:+.2f} (p={v['p']:.3f})" for k, v in comparisons[arm].items())
            print(f"  {arm:{width}s} {line}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"agents": agents, "arms": arms, "comparisons": comparisons}, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
