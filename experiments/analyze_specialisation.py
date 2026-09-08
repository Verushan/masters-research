"""How much of the achievable partner-specialisation do our agents actually use?

`analyze_partner_effects.py` reports an adaptation index -- how far an agent's
behaviour profile moves as its partner changes -- but the number has no scale.
0.30 is not obviously large or small, because nothing says how much an agent
*could* move if it were specialising perfectly.

The best-response agents supply that scale. Each was trained against exactly one
frozen partner, so its behaviour is the optimal response to that partner and
nothing else. The spread of BR profiles across partners is therefore the
maximum that partner-specialisation is worth on this layout: if the best
responses to sixteen different partners all look the same, there is nothing to
specialise about, and an agent that ignores its partner is losing nothing.

    specialisation ratio = agent's adaptation index / BR spread

Near 0, the agent plays one fixed role regardless of partner. Near 1, it moves
as much as the optimal response does. The denominator is what makes a
difference between conditions -- `_m` against `_mx`, say -- readable as a
fraction of what was available rather than as an uncalibrated delta.

Two things this deliberately does not claim:

* The BR profiles come from different *agents*, one per partner, while an
  agent's adaptation index comes from one agent across partners. The BR spread
  bounds how much an adaptive policy *should* vary; it is not something a
  single policy is guaranteed to be able to reach.
* A BR trained at 2e6 steps is a lower bound on the true best response, so the
  denominator is "the best response we could find" and the ratio is, if
  anything, generous to our agents.

    python experiments/analyze_specialisation.py --layout unident_s \
        --metrics experiments/results/recovered/metrics_unident_s_s2_hsp.json \
        --records experiments/results/recovered/cross_play_unident_s_s2_hsp_deterministic.json.gz
"""

import argparse
import gzip
import json
import os
import pathlib
import re
from collections import defaultdict

import numpy as np

# Task events only. STAY and MOVEMENT are deliberately excluded: MOVEMENT fires
# on almost every step (~300 an episode against 2-12 for any task event), so a
# profile normalised over all counters is ~66% MOVEMENT and both the ceiling and
# the adaptation index end up measuring how much an agent walked rather than
# which part of the job it did. Including them made the best responses look
# nearly identical to each other (spread 0.062) while our agents looked wildly
# variable (0.18-0.31) -- an artefact of locomotion share, not a finding about
# specialisation.
EVENTS = [
    "pickup_onion_from_O", "pickup_onion_from_X", "put_onion_on_X",
    "PLACEMENT_IN_POT", "pickup_dish_from_D", "put_dish_on_X",
    "USEFUL_DISH_PICKUP", "SOUP_PICKUP", "put_soup_on_X",
    "pickup_soup_from_X", "delivery",
]
# br_agent-hsp{i}_{tag}_w0-eval_ep_{event}_by_agent{n}
BR_KEY = re.compile(
    r"^br_agent-hsp(\d+)_\w+?_w0-eval_ep_(?P<event>.+)_by_agent(?P<side>[01])$"
)


def load_env():
    for line in (
        pathlib.Path("/home/elementrix/coding/masters-research/.env")
        .read_text()
        .splitlines()
    ):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()


def load_records(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        p = json.load(f)
    return p["records"] if isinstance(p, dict) else p


def normalise(counts):
    v = np.array([counts.get(e, 0.0) for e in EVENTS], float)
    t = v.sum()
    return v / t if t > 0 else v


def br_profiles(layout):
    """partner index -> the BR's own behaviour profile against that partner."""
    import wandb

    api = wandb.Api(timeout=60)
    ent, proj = os.environ["WANDB_ENTITY"], os.environ["WANDB_PROJECT"]
    best = {}
    for r in api.runs(f"{ent}/{proj}", {"config.layout_name": layout}, per_page=400):
        if r.config.get("experiment_name") != "br" or r.state != "finished":
            continue
        by_side = defaultdict(dict)
        partner = None
        for key, value in r.summary.items():
            m = BR_KEY.match(key)
            if not m or not isinstance(value, (int, float)):
                continue
            partner = int(m.group(1))
            by_side[int(m.group("side"))][m.group("event")] = float(value)
        if partner is None:
            continue
        # The BR is the trained seat. Identify it by activity rather than by
        # position: the frozen partner can sit in either seat depending on how
        # the pool was ordered, and guessing wrong inverts every profile.
        side = max(by_side, key=lambda s: sum(by_side[s].values()))
        score = sum(by_side[side].values())
        # One partner can have more than one finished BR run; keep the busiest,
        # which is the one that actually learned to play.
        if partner not in best or score > best[partner][1]:
            best[partner] = (normalise(by_side[side]), score)
    return {k: v[0] for k, v in best.items()}


def agent_profiles(metrics, records):
    """ego -> {partner -> profile}, folding both seating orders."""
    partners = set(metrics["partners"])
    acc = defaultdict(lambda: defaultdict(list))
    for rec in records:
        for ego, par, es in (
            (rec["agent0"], rec["agent1"], 0),
            (rec["agent1"], rec["agent0"], 1),
        ):
            if par in partners and ego not in partners:
                counts = {
                    e: rec.get(f"eval_ep_{e}_by_agent{es}", 0.0) for e in EVENTS
                }
                acc[ego][par].append(normalise(counts))
    return {
        ego: {p: np.mean(v, axis=0) for p, v in d.items()} for ego, d in acc.items()
    }


def spread(profiles):
    """Mean distance of each profile from the set's centre."""
    if len(profiles) < 2:
        return float("nan")
    a = np.array(profiles)
    return float(np.linalg.norm(a - a.mean(0), axis=1).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", default="unident_s")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--records", required=True)
    args = ap.parse_args()

    load_env()
    metrics = json.load(open(args.metrics))
    records = load_records(args.records)

    br = br_profiles(args.layout)
    ceiling = spread(list(br.values()))
    print("=" * 74)
    print(f"partner specialisation -- {args.layout}")
    print("=" * 74)
    print(
        f"\nbest-response spread over {len(br)} partners: {ceiling:.3f}"
        "\n  this is how much the optimal response itself changes with the partner,"
        "\n  so it is the most an adaptive policy could usefully vary."
    )
    if not np.isfinite(ceiling) or ceiling <= 0:
        print("  no usable ceiling; cannot calibrate")
        return

    pa = metrics["per_agent"]
    by_group = defaultdict(list)
    for ego, d in agent_profiles(metrics, records).items():
        if ego in pa and len(d) >= 3:
            by_group[pa[ego]["group"]].append(spread(list(d.values())))

    print(f"\n{'group':26s} {'n':>2s} {'adaptation':>11s} {'% of ceiling':>13s}")
    print("-" * 74)
    for g in sorted(by_group, key=lambda x: -np.mean(by_group[x])):
        m = float(np.mean(by_group[g]))
        print(f"{g:26s} {len(by_group[g]):2d} {m:11.3f} {m / ceiling:12.0%}")
    print(
        "\nOver 100% does NOT mean over-adapting. It means behaviour moves more across\n"
        "partners than the optimal response does, which is what being pushed around by\n"
        "the partner looks like: a competent best response imposes a stable joint\n"
        "policy, an incompetent agent gets dragged. Magnitude cannot tell those apart,\n"
        "which is what the next measure is for."
    )

    # Direction, not magnitude. When paired with partner p, is the ego's
    # behaviour closer to the best response *to p* than to the best responses to
    # the other partners? Only that makes the variation adaptive -- drift of any
    # size is worth nothing if it does not point at the right response.
    print("\n" + "=" * 74)
    print("is the variation in the right direction?")
    print("=" * 74)
    print(f"\n{'group':26s} {'n':>2s} {'alignment':>10s}")
    print("-" * 74)
    aligned = defaultdict(list)
    partner_index = {}
    for name in metrics["partners"]:
        hit = re.search(r"hsp(\d+)", name)
        if hit:
            partner_index[name] = int(hit.group(1))

    for ego, d in agent_profiles(metrics, records).items():
        if ego not in pa:
            continue
        scores = []
        for par, prof in d.items():
            idx = partner_index.get(par)
            if idx is None or idx not in br:
                continue
            own = float(np.linalg.norm(prof - br[idx]))
            others = [float(np.linalg.norm(prof - br[j])) for j in br if j != idx]
            if not others:
                continue
            other = float(np.mean(others))
            if own + other > 0:
                scores.append((other - own) / (other + own))
        if len(scores) >= 3:
            aligned[pa[ego]["group"]].append(float(np.mean(scores)))

    # Null: the same distances with the partner labels shuffled. Alignment is a
    # ratio of distances between profiles that are not independent, so "greater
    # than zero" is not automatically meaningful -- if every ego and every BR
    # cluster in the same region, mismatched pairs can look aligned too. The
    # shuffle keeps the geometry and destroys only the pairing.
    rng = np.random.default_rng(0)
    null = []
    profiles = agent_profiles(metrics, records)
    br_keys = sorted(br)
    for _ in range(200):
        perm = list(rng.permutation(br_keys))
        mapping = dict(zip(br_keys, perm))
        for ego, d in profiles.items():
            if ego not in pa:
                continue
            scores = []
            for par, prof in d.items():
                idx = partner_index.get(par)
                if idx is None or idx not in br:
                    continue
                fake = mapping[idx]
                own = float(np.linalg.norm(prof - br[fake]))
                others = [
                    float(np.linalg.norm(prof - br[j])) for j in br if j != fake
                ]
                if others:
                    other = float(np.mean(others))
                    if own + other > 0:
                        scores.append((other - own) / (other + own))
            if len(scores) >= 3:
                null.append(float(np.mean(scores)))
    null = np.array(null)
    lo, hi = np.percentile(null, [2.5, 97.5])

    for g in sorted(aligned, key=lambda x: -np.mean(aligned[x])):
        v = aligned[g]
        m = float(np.mean(v))
        mark = " *" if m > hi else ("  " if m > lo else " (below null)")
        print(f"{g:26s} {len(v):2d} {m:+10.3f}{mark}")
    print(
        f"\nshuffled-partner null: mean {null.mean():+.3f}, 95% range "
        f"[{lo:+.3f}, {hi:+.3f}] over {len(null)} draws"
        "\n* marks a group above the null's 97.5th percentile."
    )
    print(
        "\n> 0 the ego sits nearer the best response to its actual partner than to the\n"
        "    best responses to other partners -- the variation is adaptive.\n"
        "~ 0 the ego is no nearer the right response than to any other; whatever it is\n"
        "    doing differently with this partner, it is not moving toward the response\n"
        "    that partner calls for."
    )


if __name__ == "__main__":
    main()
