"""True BR-Prox: how wrong was the pool-max proxy?

`analyze_crossplay.py` approximates each partner's best response with the best
score any agent *already in the pool* achieved against it. Nothing in the pool
was trained to exploit anyone, so that is biased low, and every quantity built
on it is optimistic in the same direction: BR-Prox overstates how much of the
achievable return a method captures, regret understates the shortfall, and CVaR
understates the worst case.

`br-partners.slurm` trains an actual best response per held-out partner -- one
run each, against that single frozen partner -- so the denominator can be a
response rather than an accident. This reads those runs back and reports:

  * the proxy's bias per partner, and whether it is uniform. A uniform bias
    would rescale every BR-Prox number by a constant and leave the *ranking*
    intact; a non-uniform one reorders which partners are hard, which is what
    worst-case and CVaR are defined over.
  * BR-Prox recomputed against the trained denominator.

A BR trained at 2e6 steps is itself only a lower bound on the true best
response, so this is "proximity to the best response we could find". That is
still far tighter than the pool max, and the report should say which it is.

    python experiments/analyze_br_prox.py --layouts random0 unident_s random3
"""

import argparse
import json
import os
import pathlib
import re
from collections import defaultdict

import numpy as np

# What br-partners.slurm logs: one metric per (br_agent, frozen partner) pair.
BR_METRIC = re.compile(r"^br_agent-hsp(\d+)_(\w+?)_w0-ep_sparse_r$")

METRICS_FOR = {
    "random0": "experiments/results/recovered/metrics_random0_s2_hsp.json",
    "unident_s": "experiments/results/recovered/metrics_unident_s_s2_hsp.json",
    "random3": "experiments/results/metrics_random3_hsp.json",
}


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


def trained_br(api, ent, proj, layout):
    """Partner index -> return of the BR trained against it."""
    out = {}
    for r in api.runs(f"{ent}/{proj}", {"config.layout_name": layout}, per_page=400):
        if r.config.get("experiment_name") != "br" or r.state != "finished":
            continue
        for key, value in r.summary.items():
            m = BR_METRIC.match(key)
            if m and isinstance(value, (int, float)):
                # The seed is the partner index, so a run reports one pair.
                out[int(m.group(1))] = float(value)
    return out


def proxy_br(path):
    """Partner index -> br_hat, the pool-max proxy."""
    if not os.path.exists(path):
        return {}
    m = json.load(open(path))
    out = {}
    for name, value in (m.get("br_hat") or {}).items():
        hit = re.search(r"hsp(\d+)", name)
        if hit:
            out[int(hit.group(1))] = float(value)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layouts", nargs="+", default=["random0", "unident_s", "random3"])
    args = ap.parse_args()

    load_env()
    import wandb

    api = wandb.Api(timeout=60)
    ent, proj = os.environ["WANDB_ENTITY"], os.environ["WANDB_PROJECT"]

    for layout in args.layouts:
        trained = trained_br(api, ent, proj, layout)
        proxy = proxy_br(METRICS_FOR.get(layout, ""))
        shared = sorted(set(trained) & set(proxy))

        print("=" * 76)
        print(f"{layout}: {len(trained)} trained BRs, {len(proxy)} proxy values, "
              f"{len(shared)} comparable")
        print("=" * 76)
        if not trained:
            print("  no BR runs found\n")
            continue
        if not shared:
            print("  no overlap with a metrics file; trained BRs only:")
            for i in sorted(trained):
                print(f"    hsp{i:<3d} BR {trained[i]:7.1f}")
            print()
            continue

        print(f"{'partner':10s} {'proxy':>8s} {'trained BR':>11s} {'proxy/BR':>9s}")
        print("-" * 76)
        ratios = []
        for i in shared:
            p, b = proxy[i], trained[i]
            ratio = p / b if b > 0 else float("nan")
            ratios.append(ratio)
            print(f"hsp{i:<7d} {p:8.1f} {b:11.1f} {ratio:9.2f}")

        # The honest denominator is the best response found by ANY means. A
        # trained BR is a lower bound on the true one, and on some partners it
        # is a *worse* lower bound than the pool max -- 6 of 16 on unident_s,
        # and most of random0, where BR training largely failed to learn at all.
        # Taking the max of the two is never worse than either and is what
        # "proximity to the best response we could find" actually means.
        combined = {i: max(proxy[i], trained[i]) for i in shared}
        gained = sum(1 for i in shared if trained[i] > proxy[i])
        lost = sum(1 for i in shared if trained[i] < proxy[i])
        print(
            f"\n  combined denominator max(pool, trained): trained wins on "
            f"{gained}/{len(shared)}, pool max still wins on {lost}"
        )
        zero = [i for i in shared if combined[i] <= 0]
        if zero:
            print(f"  still no response found for partners {zero}")

        r = np.array([x for x in ratios if not np.isnan(x)])
        if len(r):
            print(
                f"\n  proxy captures {r.mean():.0%} of the trained BR on average "
                f"(range {r.min():.0%}-{r.max():.0%}, sd {r.std(ddof=1) if len(r) > 1 else 0:.2f})"
            )
            if len(r) > 1 and r.std(ddof=1) > 0.10:
                print(
                    "  the bias is NOT uniform, so it does not merely rescale BR-Prox --\n"
                    "  it reorders which partners look hard, which is what worst-case\n"
                    "  and CVaR are defined over."
                )
        # Does the ranking of partner difficulty survive?
        if len(shared) > 2:
            from scipy.stats import spearmanr

            rho, p_val = spearmanr([proxy[i] for i in shared], [trained[i] for i in shared])
            print(
                f"  partner-difficulty ranking, proxy vs trained: Spearman "
                f"{rho:+.2f} (p={p_val:.3f})"
            )
        print()


if __name__ == "__main__":
    main()
