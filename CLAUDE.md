# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A thin research wrapper around **ZSC-Eval** (zero-shot coordination benchmark for Overcooked / Google Research Football). The repo itself contains only orchestration:

- `pipelines/*.slurm` — end-to-end experiment pipelines (train → extract → generate ymls → train stage 2 → extract)
- `util/sync-wandb.slurm` — bulk `wandb sync` of offline run dirs under `$HOME/ZSC`
- `.env` — the single source of paths/threads/W&B config for every script (gitignored)
- `zsc-eval/` — git submodule: a **fork** of `sjtu-marl/ZSC-Eval` (`git@github.com:Verushan/zsc-eval.git`, branch `research-branch`). All algorithm/env/runner code lives here.

Nearly all real code changes happen inside the submodule. Commit there first, then commit the updated submodule pointer in the parent repo. `zsc-eval` tracks `research-branch`, not `master`; `origin/master` in the submodule is the upstream code and is useful as a diff baseline (`git diff origin/master...HEAD`).

## Environment

Conda env `zsceval` (python 3.9, pytorch + cuda 11.8) created from `zsc-eval/environment.yml`; `pip install -e .` is wired via `requirements.txt`.

Every script loads `../.env` relative to its own directory, so **run pipeline scripts from the directory they live in** (`pipelines/`, `util/`). Key variables:

| Var | Meaning |
| --- | --- |
| `PYTHONPATH` | Absolute path to `zsc-eval/`. Also used as the results root (`$PYTHONPATH/results/...`) via `zsceval/utils/train_util.py:get_base_run_dir` and `eval/eval.py`. |
| `POLICY_POOL` | Absolute path to `zsc-eval/zsceval/scripts/overcooked/policy_pool`. Every yml/model path is resolved relative to this. |
| `ROLLOUT_THREADS` / `TRAINING_THREADS` | Passed as `--n_rollout_threads` / `--n_training_threads` by the shell scripts. |
| `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_MODE`, `WANDB_API_KEY` | W&B; `--wandb_name $WANDB_ENTITY` is passed to training. Model extraction reads `WANDB_ENTITY` directly. |
| `SHOULD_SOURCE_CONDA` | `true` on the cluster (sources `~/anaconda3`), `false` locally. |

## Common commands

```bash
# Full FCP pipeline (SLURM cluster)
cd pipelines && sbatch sp-run.slurm
# ...or run the same pipeline locally
cd pipelines && bash sp-run.slurm

# Render/eval two named policies against each other
cd pipelines && bash eval-run.slurm

# Sync offline W&B runs
cd util && sbatch sync-wandb.slurm
```

Individual stages (from `zsc-eval/zsceval/scripts/`, with `.env` exported):

```bash
cd $PYTHONPATH/zsceval/scripts/overcooked
bash shell/train_sp.sh random0                     # stage 1: self-play seeds
cd .. && python extract_models/extract_sp_models.py random0 overcooked
python prep/gen_S2_yml.py random0 fcp              # build s2 population ymls
cd overcooked && bash shell/train_fcp_stage_2.sh random0 16
cd .. && python extract_models/extract_S2_models.py --layout random0 --env overcooked --algorithm fcp
```

There is no test suite and no lint target. The submodule has a `.pre-commit-config.yaml` (black at line-length 120, isort, autoflake) but the fork's files have been reformatted at black's default 88 columns — running pre-commit would rewrite large parts of the diff, so don't run it opportunistically.

## Pipeline architecture

Two-stage population training (FCP is the worked example; MEP/TrajeDi/HSP/COLE/E3T follow the same shape):

1. **Stage 1** — `shell/train_sp.sh` trains N self-play agents, one W&B run per seed, checkpointing `actor_periodic_*.pt` as W&B files.
2. **Extraction** — `extract_models/extract_sp_models.py` queries the **W&B API** (not the local filesystem) for finished runs matching `experiment_name` + layout, picks `init`/`mid`/`final` checkpoints by interpolated `ep_sparse_r`, and writes them to `$POLICY_POOL/{layout}/fcp/s1/{exp}/sp{seed}_{tag}_actor.pt`. Runs must be **finished and synced** or nothing is extracted.
3. **Yml generation** — `prep/gen_S2_yml.py {layout} fcp` writes `$POLICY_POOL/{layout}/fcp/s2/train-s{POP_SIZE}-sp-{1..5}.yml`, listing 3 checkpoints per stage-1 agent plus the trainable `fcp_adaptive` policy.
4. **Stage 2** — `shell/train_fcp_stage_2.sh {layout} {population_size}` runs `train/train_adaptive.py` against those ymls with `--population_size $((population_size * 3))`.
5. **Extraction** — `extract_models/extract_S2_models.py --layout --env --algorithm` (note: flags, not the positional args shown in the upstream README).
6. **Evaluation / render** — `eval/eval.py` with `--algorithm_name population`, a population yml, and `--agent0_policy_name` / `--agent1_policy_name` naming entries in that yml; results land in `$PYTHONPATH/results/{layout}/*.json`. `prep/eval_sp_yml.py` (fork-only) builds a two-policy yml from raw `.pt` paths for ad-hoc head-to-head evals.

**Size coupling that breaks silently-ish:** `train_sp.sh`'s seed range must produce as many stage-1 agents as `TOTAL_SIZE_LIST` in `prep/gen_S2_yml.py` expects (currently `TOTAL_SIZE=20`, `POP_SIZE=16`), and the `population_size` argument to `train_fcp_stage_2.sh` must equal `POP_SIZE` so the yml filename resolves. `gen_S2_yml.py` asserts on a missing `.pt`; a mismatched population size instead fails later with a missing yml. `extract_sp_models.py` also has a hardcoded `exp_names = {"random0": "sp", "random3_m": "sp"}` map — a new layout must be added there.

Layout name determines the env version: the six "old" layouts (`random0`, `random0_medium`, `random1`, `random3`, `small_corridor`, `unident_s`) use `zsceval/envs/overcooked/`; `*_m` multi-recipe layouts use `zsceval/envs/overcooked_new/`. The shell scripts derive `--overcooked_version` from this list.

## Fork-specific deviations from the upstream README

The upstream `zsc-eval/README.md` is still the best reference for the algorithm-level workflow, but it is stale in these respects:

- Paths come from `PYTHONPATH` / `POLICY_POOL` env vars instead of hardcoded `~/ZSC` and relative `policy_pool`.
- Thread counts and rollout counts come from `.env`, not from literals in the shell scripts.
- `--dummy_batch_size` and `--n_eval_rollout_threads` default to the largest factor of `--n_rollout_threads` ≤ 10 (`calculate_dummy_chunk_length` in `zsceval/config.py`) so the `nenvs % dummy_batch_size == 0` assertion holds for any thread count.
- Added `--wandb_tags` and `--data_parallel` (multi-GPU; `assert n_gpu == 1 or all_args.data_parallel` in the train scripts).
- `print` replaced with `loguru` logging throughout runners/extractors.
- `num_env_steps` / `reward_shaping_horizon` in `train_sp.sh` and the `population_size == 16` branch of `train_fcp_stage_2.sh` are **reduced short-run values** (1e6 / 5e5) for pipeline shakeout, not the paper's 1e7–1e8. Restore them before producing real results.
