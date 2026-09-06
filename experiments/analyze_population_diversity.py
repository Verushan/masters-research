"""Is a MORL stage-1 population more behaviourally diverse than a shaped one?

The stage-2 result raised a mechanism question the stage-1 analysis could not
answer. `analyze_partner_effects.py` falsified the obvious explanation -- MORL
agents are not measurably more partner-responsive than shaped ones, their
adaptation indices sit within noise of each other -- yet a population of MORL
self-play agents trained a stage-2 agent that scored differently from one
trained on a shaped population. If the individuals are not the cause, the
population might be: FCP-style training wants partners that differ from each
other, and an arm whose members are more spread out is a harder and more
informative curriculum regardless of how good any single member is.

This measures that spread directly, from cross-play data already on disk. No
new rollouts.

Two notions of diversity, because they can disagree and the disagreement is
informative:

behavioural  mean pairwise L2 distance between members' L1-normalised event
             profiles. Answers "do these agents play differently from one
             another", independently of how well they score.
outcome      mean pairwise L2 distance between members' return vectors against
             the held-out partner set, each normalised by the pool's return
             scale. Answers "do these agents succeed and fail against
             *different* partners" -- the property a curriculum actually needs,
             since two agents that play differently but fail against exactly
             the same partners teach the ego agent the same lesson.

Both are computed over an arm's members as they appear in the cross-play
matrix, folding the two seating orders together the way `cross_play.py` writes
them.

    python experiments/analyze_population_diversity.py \
        --metrics experiments/results/metrics_unident_s_s2_hsp.json \
        --crossplay experiments/results/cross_play_unident_s_s2_hsp_deterministic.json.gz

Interpretation warning, stated here because the number is easy to over-read:
there are four stage-1 arms per layout, so any correlation between diversity
and stage-2 score is computed over a handful of points and is descriptive
only. It can rule a mechanism *out* (no spread difference at all means
diversity cannot be the explanation); it cannot establish one.
"""

import argparse
import gzip
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

BEHAVIOUR_EVENTS = [
    "pickup_onion_from_O",
    "pickup_onion_from_X",
    "put_onion_on_X",
    "PLACEMENT_IN_POT",
    "pickup_dish_from_D",
    "put_dish_on_X",
    "USEFUL_DISH_PICKUP",
    "SOUP_PICKUP",
    "put_soup_on_X",
    "pickup_soup_from_X",
    "delivery",
    "STAY",
    "MOVEMENT",
]

# Stage-1 arms. `_peak` variants are the same seeds at a different checkpoint,
# so including them would count each seed twice and shrink the spread.
STAGE1_ARMS = ["bench_sp", "bench_sparse", "bench_morl", "bench_morl_ad"]


def load_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def episode_profile(record, side):
    counts = np.array(
        [record.get(f"eval_ep_{e}_by_agent{side}", 0.0) for e in BEHAVIOUR_EVENTS],
        dtype=np.float64,
    )
    total = counts.sum()
    return counts / total if total > 0 else counts


def member_profiles(records, members):
    """Mean behaviour profile per agent, over every episode and both seats."""
    wanted = set(members)
    rows = defaultdict(list)
    for record in records:
        for agent, side in ((record["agent0"], 0), (record["agent1"], 1)):
            if agent in wanted:
                rows[agent].append(episode_profile(record, side))
    return {a: np.mean(v, axis=0) for a, v in rows.items() if v}


def mean_pairwise_distance(vectors):
    if len(vectors) < 2:
        return float("nan")
    return float(
        np.mean([np.linalg.norm(a - b) for a, b in combinations(vectors, 2)])
    )


def outcome_vectors(metrics, members, partners):
    """Each member's return profile across the held-out partners."""
    out = {}
    for m in members:
        vs = metrics["per_agent"][m]["vs_partners"]
        out[m] = np.array([vs.get(p, np.nan) for p in partners], dtype=np.float64)
    return out


def analyse(metrics_path, crossplay_path, label=None):
    """Print one layout's table and return its (arm, behavioural, outcome, zsc, s2) rows."""

    class _A:
        pass

    args = _A()
    args.metrics, args.crossplay, args.label = metrics_path, crossplay_path, label
    return _run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--crossplay", required=True)
    parser.add_argument("--label", default=None, help="Layout name for the header.")
    parser.add_argument(
        "--also",
        nargs=3,
        action="append",
        metavar=("METRICS", "CROSSPLAY", "LABEL"),
        default=[],
        help="A second layout to analyse and rank alongside the first. Repeatable. "
        "Stage-2 returns are not comparable across layouts -- random0 scores an "
        "order of magnitude below unident_s -- so the pooled view ranks arms "
        "*within* each layout and asks whether the ordering agrees, rather than "
        "correlating raw returns across them.",
    )
    args = parser.parse_args()
    rows = {(args.label or Path(args.metrics).stem): _run(args)}
    for m, c, lab in args.also:
        rows[lab] = analyse(m, c, lab)
    if len(rows) > 1:
        rank_across_layouts(rows)


def rank_across_layouts(rows_by_layout):
    """Do the arms rank the same way on diversity as on stage-2 score?"""
    print("=" * 78)
    print("within-layout ranking (stage-2 returns are not comparable across layouts)")
    print("=" * 78)
    agree = []
    for layout, rows in rows_by_layout.items():
        rows = [r for r in rows if not np.isnan(r[4])]
        if len(rows) < 2:
            continue
        by_div = sorted(rows, key=lambda r: r[2], reverse=True)
        by_s2 = sorted(rows, key=lambda r: r[4], reverse=True)
        print(f"\n{layout}")
        print(f"  by outcome diversity: {' > '.join(r[0] for r in by_div)}")
        print(f"  by stage-2 ZSC      : {' > '.join(r[0] for r in by_s2)}")
        top_match = by_div[0][0] == by_s2[0][0]
        agree.append(top_match)
        print(
            f"  most-diverse arm is also the best stage-2 arm: "
            f"{'YES' if top_match else 'no'}  ({by_div[0][0]})"
        )
    if agree:
        print(
            f"\n  top-arm agreement: {sum(agree)}/{len(agree)} layouts.\n"
            "  If this holds up, the predictor of a good stage-2 agent is the\n"
            "  population's outcome diversity rather than whether its reward was\n"
            "  MORL or hand-shaped -- which is what makes the two layouts\n"
            "  disagree about MORL while agreeing about diversity."
        )
    print()


def _run(args):
    metrics = load_json(args.metrics)
    payload = load_json(args.crossplay)
    records = payload["records"] if isinstance(payload, dict) else payload

    per_agent = metrics["per_agent"]
    partners = sorted(metrics["partners"])
    label = args.label or Path(args.metrics).stem

    print("=" * 78)
    print(f"population diversity -- {label}")
    print(f"partner group: {metrics['partner_group']}  ({len(partners)} partners)")
    print("=" * 78)

    # Return scale: the pool's own spread, so the outcome distance is comparable
    # across layouts whose absolute returns differ by an order of magnitude.
    all_returns = np.array(
        [v for a in per_agent.values() for v in a["vs_partners"].values()],
        dtype=np.float64,
    )
    scale = np.nanstd(all_returns) or 1.0

    print(
        f"\n{'arm':16s} {'n':>2s} {'behavioural':>12s} {'outcome':>9s} "
        f"{'ZSC mean':>9s} {'s2 ZSC':>8s}"
    )
    print("-" * 78)

    rows = []
    for arm in STAGE1_ARMS:
        members = sorted(a for a, v in per_agent.items() if v["group"] == arm)
        if len(members) < 2:
            continue
        profiles = member_profiles(records, members)
        beh = mean_pairwise_distance([profiles[m] for m in members if m in profiles])
        outs = outcome_vectors(metrics, members, partners)
        out = mean_pairwise_distance([outs[m] / scale for m in members])
        zsc = float(np.mean([per_agent[m]["zsc_mean"] for m in members]))

        s2_members = [a for a, v in per_agent.items() if v["group"] == f"s2_{arm}"]
        s2 = (
            float(np.mean([per_agent[m]["zsc_mean"] for m in s2_members]))
            if s2_members
            else float("nan")
        )
        print(
            f"{arm:16s} {len(members):2d} {beh:12.4f} {out:9.4f} {zsc:9.1f} {s2:8.1f}"
        )
        rows.append((arm, beh, out, zsc, s2))

    paired = [r for r in rows if not np.isnan(r[4])]
    if len(paired) >= 3:
        beh = np.array([r[1] for r in paired])
        out = np.array([r[2] for r in paired])
        s2 = np.array([r[4] for r in paired])
        print(
            f"\n  over {len(paired)} arms with a stage-2 agent:\n"
            f"    corr(behavioural diversity, stage-2 ZSC) = {np.corrcoef(beh, s2)[0, 1]:+.3f}\n"
            f"    corr(outcome diversity,     stage-2 ZSC) = {np.corrcoef(out, s2)[0, 1]:+.3f}"
        )
        print(
            "    (descriptive only -- this is a correlation over a handful of "
            "arms,\n     reportable as a direction to check, never as evidence "
            "on its own)"
        )
    print()
    return rows


if __name__ == "__main__":
    main()
