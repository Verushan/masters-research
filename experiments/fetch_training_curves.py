#!/usr/bin/env python
"""Pull the MORL benchmark's stage-1 training curves out of W&B into one JSON.

The four arms log the same keys -- `ep_sparse_r`, the per-objective breakdown
`ep_obj_*`, and (for the arms whose reward *is* the objective vector)
`ep_morl_r` and the live preference weights `ep_morl_w_*`. Collecting them once
keeps the analysis and the report reproducible without re-querying the API.

Usage:
    python experiments/fetch_training_curves.py --layout random0 \
        --arms bench_sp bench_sparse bench_morl bench_morl_ad
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
                    ]
                },
                order="+config.seed",
            )
        )
        logger.info(f"{arm}: {len(runs)} finished runs")
        out[arm] = {}
        for run in runs:
            history = run.history(samples=2000)
            present = [k for k in KEYS if k in history]
            series = {}
            for key in present:
                # Rows logged by the eval pass leave the training keys NaN; drop
                # them per-key rather than per-row so nothing else is lost.
                sub = history[["_step", key]].dropna()
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
