"""Which objectives can be farmed without doing the task?

An objective is *farmable* when an agent can drive it up while delivering
nothing. That is not a hypothetical: on random0 the `coordination` objective
counts counter handoffs, an agent can put an onion down and pick it up forever,
and six stage-1 seeds across two arms independently found it. The scalarised
reward rises the whole time, so nothing in the training curve looks wrong.

The diagnostic needs no new rollouts, because the training runs already contain
the natural experiment. Some seeds delivered and some collapsed to zero sparse
return, and both logged the same per-objective breakdown. So for each objective:

    farmed  = max over runs that delivered nothing
    earned  = median over runs that delivered
    ratio   = farmed / earned

A ratio near zero means the objective tracks task progress -- you cannot get it
without cooking. A ratio above 1 means the objective is *better farmed than
earned*, and any weight placed on it is an invitation.

The structural reason, which is what generalises beyond this layout:

    an objective is farmable when its triggering event is reversible at no cost

`put_onion_on_X` / `pickup_onion_from_X` undo each other and advance nothing, so
the pair can be repeated without limit. `PLACEMENT_IN_POT` consumes an onion
into a pot that must then cook; `SOUP_PICKUP` requires a soup to exist. Those
are gated by irreversible progress and cannot be farmed however much weight
they carry.

That gives a design rule that is checkable *without knowing the solution*, which
is the property hand-shaped rewards lack. A shaped reward is tuned per layout
until the agent behaves; it encodes the answer. A structural criterion --
"count only events that consume a resource or advance state that cannot be
undone" -- is layout-independent, and leaves the layout-specific part to the
preference weights, which is the thing MORL is supposed to be adapting.

    python experiments/audit_objectives.py --layouts random0 unident_s
"""

import argparse
import os
import pathlib
from collections import defaultdict

import numpy as np

ARMS = [
    "bench_sp",
    "bench_sparse",
    "bench_morl",
    "bench_morl_ad",
    "bench_morl_div",
]
# Below this an episode has effectively delivered nothing.
ZERO_SPARSE = 1e-6


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layouts", nargs="+", default=["random0", "unident_s"])
    args = ap.parse_args()

    load_env()
    import wandb

    api = wandb.Api(timeout=60)
    ent, proj = os.environ["WANDB_ENTITY"], os.environ["WANDB_PROJECT"]

    for layout in args.layouts:
        runs = [
            r
            for r in api.runs(f"{ent}/{proj}", {"config.layout_name": layout}, per_page=400)
            if r.config.get("experiment_name") in ARMS and r.state == "finished"
        ]
        farmed = defaultdict(list)
        earned = defaultdict(list)
        n_zero = n_ok = 0
        for r in runs:
            s = r.summary
            sparse = s.get("ep_sparse_r")
            if sparse is None:
                continue
            objs = {
                k.replace("ep_obj_", ""): v
                for k, v in s.items()
                if k.startswith("ep_obj_") and isinstance(v, (int, float))
            }
            if not objs:
                continue
            bucket = farmed if sparse <= ZERO_SPARSE else earned
            if sparse <= ZERO_SPARSE:
                n_zero += 1
            else:
                n_ok += 1
            for k, v in objs.items():
                bucket[k].append(float(v))

        print("=" * 76)
        print(f"{layout}: {n_ok} runs delivered, {n_zero} delivered nothing")
        print("=" * 76)
        if not farmed or not earned:
            print("  need runs of both kinds to compute a ratio\n")
            continue

        print(
            f"{'objective':20s} {'farmed(max)':>12s} {'earned(med)':>12s} "
            f"{'ratio':>7s}  verdict"
        )
        print("-" * 76)
        for k in sorted(set(farmed) | set(earned)):
            f = max(farmed.get(k, [0.0]))
            e = float(np.median(earned.get(k, [0.0])))
            ratio = f / e if e > 0 else (float("inf") if f > 0 else 0.0)
            if ratio >= 1.0:
                verdict = "FARMABLE -- better farmed than earned"
            elif ratio >= 0.5:
                verdict = "weak -- reachable without the task"
            else:
                verdict = "task-anchored"
            print(f"{k:20s} {f:12.1f} {e:12.1f} {ratio:7.2f}  {verdict}")
        print()


if __name__ == "__main__":
    main()
