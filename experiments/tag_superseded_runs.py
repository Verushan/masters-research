"""Mark the pre-anchoring MORL runs as superseded, without destroying them.

Three arms take the objective vector as their reward -- bench_morl,
bench_morl_ad, bench_morl_div -- and every run of them predating the anchored
coordination objective optimised a farmable reward. Those runs should not be
mistaken for results, but they must not be deleted either: they *are* the
evidence for the farming finding. `audit_objectives.py` derives the 3.69
farmed-to-earned ratio by comparing runs that delivered nothing against runs
that delivered, and both groups come from exactly this set. Deleting them makes
the result that motivated the whole correction unreproducible.

So this tags rather than removes:

    objectives-default   factual: which objective set the run optimised. New
                         runs already carry their set as a tag, so applying it
                         retroactively makes the whole history queryable the
                         same way.
    superseded           this run has been replaced by an `-anc` counterpart

bench_sp and bench_sparse are deliberately untouched. Neither takes the
objective vector as its reward -- bench_sparse is w = (20,0,0,0), task
completion only, which is not farmable -- so the baseline is unaffected and
should not be labelled as though it were.

Note the tag is *not* `unused`. That name already means something here: it is
what the stage-2 extractor filters on, and hand-applying it to resolve seed
collisions is what produced the biased subsample that cost us the original
headline. Reusing it would be the same mistake twice.

    python experiments/tag_superseded_runs.py --dry-run
    python experiments/tag_superseded_runs.py --apply
"""

import argparse
import os
import pathlib
from collections import defaultdict

MORL_ARMS = ["bench_morl", "bench_morl_ad", "bench_morl_div"]
TAGS = ["objectives-default", "superseded"]


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
    ap.add_argument("--apply", action="store_true", help="Actually write the tags.")
    ap.add_argument("--dry-run", action="store_true", default=True)
    args = ap.parse_args()

    load_env()
    import wandb

    api = wandb.Api(timeout=60)
    ent, proj = os.environ["WANDB_ENTITY"], os.environ["WANDB_PROJECT"]

    targets = []
    for layout in args.layouts:
        for r in api.runs(
            f"{ent}/{proj}", {"config.layout_name": layout}, per_page=400
        ):
            exp = r.config.get("experiment_name") or ""
            # The re-baselined runs carry a suffix; only the originals qualify.
            if exp not in MORL_ARMS:
                continue
            if r.config.get("morl_objectives") not in (None, "default"):
                continue
            targets.append((layout, exp, r))

    by_group = defaultdict(list)
    for layout, exp, r in targets:
        by_group[(layout, exp)].append(r)

    print(f"{'layout':12s} {'arm':16s} {'runs':>5s}  already tagged")
    print("-" * 60)
    total = tagged = 0
    for (layout, exp), rs in sorted(by_group.items()):
        done = sum(1 for r in rs if set(TAGS) <= set(r.tags or []))
        total += len(rs)
        tagged += done
        print(f"{layout:12s} {exp:16s} {len(rs):5d}  {done}")
    print(f"\n{total} runs match, {tagged} already carry {TAGS}")

    if not args.apply:
        print("\ndry run -- nothing written. Re-run with --apply to tag.")
        return

    changed = 0
    for _, _, r in targets:
        existing = set(r.tags or [])
        if set(TAGS) <= existing:
            continue
        r.tags = sorted(existing | set(TAGS))
        r.update()
        changed += 1
    print(f"\ntagged {changed} runs with {TAGS}")


if __name__ == "__main__":
    main()
