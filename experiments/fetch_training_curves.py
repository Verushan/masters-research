#!/usr/bin/env python
"""Pull the MORL benchmark's training curves out of W&B into one JSON.

The four arms log the same keys -- `ep_sparse_r`, the per-objective breakdown
`ep_obj_*`, and (for the arms whose reward *is* the objective vector)
`ep_morl_r` and the live preference weights `ep_morl_w_*`. Collecting them once
keeps the analysis and the report reproducible without re-querying the API.

Stage-2 runs log the same metrics, but namespaced by the policy pair they were
measured on (`either-fcp_adaptive-ep_sparse_r`), so they need `--key_prefix`;
the prefix is stripped again on the way out so both stages produce the same
series names.

Usage:
    python experiments/fetch_training_curves.py --layout random0 \
        --arms bench_sp bench_sparse bench_morl bench_morl_ad

    python experiments/fetch_training_curves.py --layout random0 \
        --arms fcp-S2-bench_sp fcp-S2-bench_morl_ad \
        --key_prefix either-fcp_adaptive- --out results/training_curves_s2_random0.json
"""

import argparse
import json
import os
from pathlib import Path

import wandb
from loguru import logger

KEYS = [
    "ep_sparse_r",
    "ep_shaped_r",
    "ep_morl_r",
    "ep_obj_task_completion",
    "ep_obj_ingredient_prep",
    "ep_obj_plating",
    "ep_obj_coordination",
    "ep_morl_w_task_completion",
    "ep_morl_w_ingredient_prep",
    "ep_morl_w_plating",
    "ep_morl_w_coordination",
    "eval_ep_sparse_r",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="random0")
    parser.add_argument(
        "--arms", nargs="+", default=["bench_sp", "bench_sparse", "bench_morl", "bench_morl_ad"]
    )
    parser.add_argument(
        "--out", default=str(Path(__file__).parent / "results" / "training_curves.json")
    )
    parser.add_argument(
        "--key_prefix",
        default="",
        help="Prefix the metrics carry in W&B, e.g. 'either-fcp_adaptive-' for "
        "stage-2 runs. Stripped from the series names in the output.",
    )
    args = parser.parse_args()

    entity = os.environ["WANDB_ENTITY"]
    project = os.environ["WANDB_PROJECT"]
    api = wandb.Api(timeout=60)

    out = {}
    for arm in args.arms:
        runs = list(
            api.runs(
                f"{entity}/{project}",
                filters={
                    "$and": [
                        {"config.experiment_name": arm},
                        {"config.layout_name": args.layout},
                        {"state": "finished"},
                        # Same exclusion extract_S2_models.py applies. Without it a
                        # superseded run -- e.g. one left by a stale copy of a train
                        # script -- is silently averaged in alongside the real one,
                        # and `out[arm][seed]` keeps whichever came last.
                        {"tags": {"$nin": ["hidden", "unused"]}},
                    ]
                },
                order="+config.seed",
            )
        )
        logger.info(f"{arm}: {len(runs)} finished runs")
        out[arm] = {}
        for run in runs:
            history = run.history(samples=2000)
            present = [k for k in KEYS if f"{args.key_prefix}{k}" in history]
            series = {}
            for key in present:
                # Rows logged by the eval pass leave the training keys NaN; drop
                # them per-key rather than per-row so nothing else is lost.
                sub = history[["_step", f"{args.key_prefix}{key}"]].dropna()
                sub.columns = ["_step", key]
                series[key] = {
                    "step": sub["_step"].astype(int).tolist(),
                    "value": sub[key].astype(float).tolist(),
                }
            out[arm][str(run.config["seed"])] = {
                "run_id": run.id,
                "runtime_s": run.summary.get("_runtime"),
                "num_env_steps": run.config.get("num_env_steps"),
                "morl_weights": run.config.get("morl_weights"),
                "use_morl": run.config.get("use_morl"),
                "morl_adaptive_weights": run.config.get("morl_adaptive_weights"),
                "series": series,
            }
            logger.info(f"  seed {run.config['seed']} ({run.id}): {len(present)} keys")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    logger.success(f"wrote {args.out}")


if __name__ == "__main__":
    main()
