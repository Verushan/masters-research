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

# MORL agent: stage-1 self-play on the objective-vector reward
cd pipelines && sbatch morl-run.slurm

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

## MORL agent

`zsceval/envs/morl/` turns Overcooked's scalar reward into a reward **vector** — `task_completion`,
`ingredient_prep`, `plating`, `coordination` (`objectives.py`, registry + presets `default` /
`task_only`) — derived entirely from the `SHAPED_INFOS` counters `resolve_interacts` already
produces, so the MDP hot path is untouched.

The `morl` agent is the `sp` agent with `w · r_vec` as its PPO reward instead of
`sparse + reward_shaping_factor * shaped`. Same train script (`train/train_sp.py`), same runner,
same hyper-parameters — only the reward differs, so `sp` and `morl` runs are directly comparable.
The sparse reward still reaches it through `task_completion`; the other three objectives are dense,
which is why no hand-crafted shaping is applied.

```bash
cd $PYTHONPATH/zsceval/scripts/overcooked
python morl/rollout_objectives.py --layout random1      # objective vector vs. the env's own accounting
python morl/check_morl_reward.py                        # the RL reward really is w . r_vec
bash shell/train_morl.sh random0                        # stage 1, experiment_name "morl"
cd .. && python extract_models/extract_sp_models.py random0 overcooked morl ep_morl_r
```

Flags (all in `zsceval/overcooked_config.py`): `--morl_objectives` alone only *tracks* the vector,
so any existing sp/fcp/mep run can log per-objective breakdowns without its reward changing;
`--use_morl` additionally makes it the reward. `--morl_weights` (default uniform `1/K`),
`--morl_reward_scale`, and `--morl_adaptive_weights` + `--morl_eta_min/max`, `--morl_weight_floor`,
`--morl_weight_update_interval` for the mirror-descent preference update
(`envs/morl/preferences.py`, proposal §4.2.3). Adaptive weights are off by default: they make the
reward non-stationary while `w` is not yet part of the observation. The update is multiplicative
and fires every env step, so per-episode movement scales with `eta * episode_length` — the `eta`
defaults assume the 400-step horizon, and the floor stops an objective being switched off entirely.

**Old layouts only.** Only `zsceval/envs/overcooked/` carries the objective layer;
`train_sp.py` asserts on `--use_morl` with `--overcooked_version new`.

`extract_sp_models.py` takes two optional positional args — `{layout} {env} [exp] [metric]` — so
MORL checkpoints are ranked by `ep_morl_r` rather than `ep_sparse_r`. Both runs also log
`ep_obj_{name}` / `eval_ep_obj_{name}` per objective and per agent.

## MORL benchmark

`pipelines/morl-benchmark.slurm` + `pipelines/morl-benchmark-eval.slurm` compare the MORL agent
against the rest of the pipeline on one layout. Four stage-1 arms differ **only** in the reward the
PPO buffer sees (`shell/train_morl_benchmark.sh <layout> <arm>`):

| arm | reward |
| --- | --- |
| `bench_sp` | `sparse + reward_shaping_factor * shaped` — the ZSC-Eval baseline |
| `bench_sparse` | sparse only, via `--use_morl --morl_weights "20,0,0,0"` |
| `bench_morl` | `w · r_vec`, uniform fixed `w` |
| `bench_morl_ad` | `w · r_vec`, mirror-descent adaptive `w` |

`--morl_objectives default` is passed to every arm, including the two whose reward it does not
touch, so all four log the same `ep_obj_*` breakdown and are directly comparable. Unlike
`train_sp.sh`, the entropy and reward-shaping horizons are scaled to `num_env_steps` — on a 2e6-step
run the upstream `0 5e6 1e7` schedule never leaves the 0.2 entropy phase, which is why the older
1e6-step `sp` runs plateau near 45 sparse while these reach ~200.

```bash
cd pipelines && bash morl-benchmark.slurm                  # train + extract 4 arms
bash morl-benchmark-eval.slurm random0 "1 2 3"             # cross-play + analysis
```

Evaluation is `eval/cross_play.py`, not `eval/eval.py`: it loads the pool once and gives each env
thread a different pairing, so a full matrix costs one process instead of one per cell, and it
writes one record per episode so mean / worst-case / variance are all derivable afterwards. The pool
comes from `prep/gen_crossplay_yml.py` (arms + FCP stage-2 agent + held-out stage-1 partners).

Two passes are run. Deterministic is the canonical ZSC-Eval number — fixed start state plus argmax
actions means a pair replays identically, so one episode per cell is the entire distribution and
repeats would be duplicate rows. Stochastic (`--eval_stochastic`) is the only pass in which repeated
rollouts of one pair differ, so it is the only one that can report return stability.

Analysis lives in `experiments/`: `fetch_training_curves.py` (W&B → JSON), `analyze_crossplay.py`
(self-play / ZSC mean / worst-case / spread / stability / BR-Prox proxy), `analyze_preferences.py`
(did the mirror-descent weights move, did they track the realised objective shares, did adapting
leave behaviour closer to balanced). Results land in `experiments/results/`; `experiments/logs/` is
gitignored.

**Gotchas that cost real time:**

- `--use_recurrent_policy` and `--use_wandb` are `action="store_false"` in `zsceval/config.py`.
  Passing them *disables* the feature. That is why `train_sp.sh` passes `--use_recurrent_policy`
  with `algorithm_name=mappo` (which asserts it is False) and why every stage-1 checkpoint is an MLP
  loaded through `mlp_policy_config.pkl`. The stage-2 `fcp_adaptive` agent *is* recurrent and needs
  `rnn_policy_config.pkl`.
- `--dummy_batch_size` is envs per worker process, not a minibatch size. Setting it equal to
  `--n_eval_rollout_threads` puts every env in one process and serialises the run.
- Per-agent episode counters (`ep_sparse_r_by_agent`, `ep_category_r_by_agent`, `ep_vec_r_by_agent`)
  are reported in **base-env player order**, while rewards are swapped into ego order. They agree
  only because `--random_index` defaults to False; `cross_play.py` asserts on it.
- `random0` is forced coordination — a counter column separates the two agents, so every onion and
  dish must be handed across it. Self-play agents converge on a private protocol and their cross-play
  sparse return collapses to 0 with anyone else, which is why the objective breakdown, not the
  return, is what discriminates methods there.

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
