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

# HSP bias agents: the held-out partner set the ZSC metric is measured against
cd pipelines && sbatch hsp-partners.slurm

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
(`envs/morl/preferences.py`, proposal §4.2.3). Adaptive weights are off by default: on their own
they make the reward non-stationary, because `w` moves mid-episode while the agent has no way to
see that it moved, so two identical observations carry different returns. The update is
multiplicative and fires every env step, so per-episode movement scales with `eta * episode_length`
— the `eta` defaults assume the 400-step horizon, and the floor stops an objective being switched
off entirely.

`--use_morl_obs_weights` restores the Markov property by appending the live `w` to the observation
and the share observation as `K` constant channels, scaled by 255 to match the rest of the `ppo`
featurisation (raw weights of order `1/K` sit two orders of magnitude below every other feature and
the network never sees them). It requires `--use_morl`, and it is **opt-in because it widens the
observation space** — a policy trained with it cannot load a checkpoint saved without it, and every
agent currently in the policy pool was trained without it. `check_morl_reward.py --only obs_weights`
covers the widths, the space/observation agreement, per-step tracking under adaptive weights, and
that `reset()` restores the target rather than leaking the previous episode's `w`. Requires the
grid (`ppo`) featurisation; `bc` features are a flat vector with no channel axis.

**Old layouts only.** Only `zsceval/envs/overcooked/` carries the objective layer;
`train_sp.py` asserts on `--use_morl` with `--overcooked_version new`.

## Partner-conditioned agents (oracle upper bound, not a ZSC method)

`--use_agent_policy_id` is upstream and feeds the partner's identity to the **centralised critic
only** — it is applied in `_gen_share_observation`, so the actor never sees it and the policy cannot
condition on its partner at execution time. `--use_agent_policy_id_obs` (fork-only) appends the
identity to the **actor's** observation as well, which is what a partner-conditioning experiment
actually needs. Each agent is shown its *partner's* id, not its own: an agent's own id is constant,
and for the trainable agent it is the `-1.0` sentinel.

**This cannot generalise zero-shot, by construction.** A held-out partner has no id the agent was
ever trained on. Its value is as an *oracle ceiling*: train with the id, evaluate against training
partners with the true id, and the gap to the real ZSC number separates partner **uncertainty**
(closed by knowing who) from partner **diversity** (bad coordination even knowing who). Reporting it
as a ZSC result would be wrong.

**The id encoding is pool-size dependent.** `policy_pool.load_population` assigns
`id = (i + 1) / num_policies`, so:

- the *same* partner has a *different* id in a pool of a different size — an agent trained against a
  stage-2 yml and evaluated in a larger cross-play pool sees every partner's id shift;
- adjacent ids mean nothing but adjacency in the yml, so the raw scalar imposes an ordinal structure
  on what is categorical.

`--agent_policy_id_obs_dim N` one-hots the id over `N` partners and fixes both. `N` **must equal the
number of entries in the population yml the ids came from**; a mismatch raises rather than silently
aliasing two partners onto one index. `0` (the default) keeps the raw scalar the critic uses.

Both this and `--use_morl_obs_weights` widen the observation space, so a policy trained with either
cannot load a checkpoint saved without it — every agent currently in the pool was trained without
both. `check_morl_reward.py --only policy_id_obs` covers the widths, the partner-not-self semantics,
the one-hot recovery, the unknown-partner case, and the width mismatch.

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

## MORL stage 2 (does the pipeline fix MORL's zero-shot deficit?)

`pipelines/morl-s2.slurm` takes each stage-1 arm's population through the rest of the FCP pipeline,
which the stage-1 benchmark could not answer: MORL self-play agents coordinate better with
themselves and worse with strangers, and repairing exactly that is what population training is for.

One stage-2 run per arm. The population is that arm's three stage-1 seeds at `init`/`mid`/`final`
(9 partners, `prep/gen_arm_S2_yml.py`), so arms are matched on population size and differ only in
the reward those partners optimised. The ego agent optimises the **task** reward in every arm —
giving the MORL arms a MORL ego reward too would confound "MORL population" with "MORL ego".
`mixed` (bench_sp + bench_morl_ad, 18 partners) is available for the reward-diversity question and
is deliberately *not* size-matched.

```bash
cd pipelines && sbatch morl-s2.slurm                       # layout arms seed_begin seed_max steps
bash morl-benchmark-eval.slurm random0 "1 2 3" "bench_sp bench_morl bench_morl_ad"
```

The eval script's third argument adds the stage-2 agents to the cross-play pool and switches the
output files to `*_s2_*`, so the stage-1-only results the current report is built from are not
overwritten. `experiments/analyze_crossplay.py` groups them as `s2_{arm}`.

- Stage-2 `num_env_steps` is **not** multiplied by population size — the `*= population_size` in
  `overcooked_runner.train_mep` is the stage-1 branch — so the default 2e6 is 2e6, matched to
  stage 1 and 4x what the existing (undertrained) `fcp-S2-s16` agent got.
- `--morl_objectives` buys no extra logging in stage 2: `evaluate_with_multi_policy` drops every
  key but `eval_ep_sparse_r` / `eval_ep_shaped_r` once `stage == 2` and wandb is on. The stage-2
  objective breakdown comes from the cross-play pass.
- Stage-2 metrics are namespaced by policy pair; the aggregate the extractor ranks on is
  `either-fcp_adaptive-ep_sparse_r`, hence `fetch_training_curves.py --key_prefix`.
- `extract_S2_models.py` now takes `--exp` (the arms are logged as `fcp-S2-{arm}`, not one of the
  hardcoded `{algo}-S2-s{size}` names), and its retry loop is bounded — it previously spun forever
  on a permanent error.

**Gotchas that cost real time:**

- `--s2_suffix` must be passed as `--s2_suffix=-pilot`, not `--s2_suffix -pilot`. Suffixes start
  with a dash by convention and argparse reads a space-separated `-pilot` as a flag, not a value.
  `morl-benchmark-eval.slurm` uses the `=` form for this reason.
- `extract_S2_models.py` writes `{seed}.pt`, so two *finished* W&B runs sharing a seed overwrite each
  other and the survivor depends on the order W&B returned them in. It now raises on duplicate seeds;
  the fix is to tag the unwanted runs `unused` in W&B (the extractor's filter already drops
  `hidden`/`unused`). This is not hypothetical — two concurrent pilot loops left `bench_sp` with four
  finished runs, two of them from a stale copy of `train_morl_stage_2.sh` with `log_interval 50`.
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

## HSP partners (the held-out set the ZSC metric is measured against)

`pipelines/hsp-partners.slurm` trains the partners the zero-shot number is computed against.
Before it, `gen_crossplay_yml.py` drew them from a pre-existing `sp` pool that only ever existed
for `random0`, so no cross-layout comparison was possible. HSP bias agents are what ZSC-Eval
actually intends: they can be produced for any layout, and reward-randomised agents are far more
behaviourally diverse than self-play checkpoints of one reward.

Each agent is a pair. `w0` is the biased policy — a reward vector drawn from the enumerated
candidate space — and `w1` is a plain sparse-reward partner trained alongside it. **Only `w0` is
worth evaluating against**; `w1` is an artefact of how the pair is trained, which is why
`gen_crossplay_yml.py` only ever references `hsp{i}_{tag}_w0_actor.pt`.

`w0` is an *enumeration*, not a random draw: `train_bias_agent.py` takes the product of the
bracketed ranges in `shell/train_bias_agents.sh`, filters to at most three non-zero bias terms, and
picks `candidates[(seed + w0_offset) % len]`. A contiguous or evenly-strided seed range therefore
aliases onto a corner of the space — `itertools.product` varies the *last* dimension fastest and
`sparse_r` is last, so any even stride pins it to one level. `seed_max` in the shell script is also
not the candidate count (176 vs an actual 52 for `random3` / `unident_s`), so seeds past the count
wrap and silently retrain duplicates.

`prep/select_hsp_seeds.py` (fork-only) does the picking instead: greedy max-min over the bias terms,
each dimension scaled to unit range first so the ±20 terms do not swamp the ±0.1 ones. It also drops
the vectors where idling out-earns cooking — but only when nothing in the vector pays to act, since
a +10 dispenser pickup sits inside the delivery loop and rescues a vector that looks idle-dominant
on the STAY term alone. Selection is deterministic, so a layout name reproduces its seed list.

| layout | candidates | usable after the idle filter |
| --- | --- | --- |
| `random0` | 30 | 28 |
| `random3` | 52 | 48 |
| `unident_s` | 52 | 48 |

```bash
cd pipelines && sbatch hsp-partners.slurm              # one array task per layout
sbatch --array=0 hsp-partners.slurm                    # random0 only
sbatch -p stampede -c 16 hsp-partners.slurm            # override partition/cores at submit time

# top up one layout with named seeds, skipping selection entirely
LAYOUTS=random3 SEEDS="41 42 50 51" sbatch --array=0 --export=ALL hsp-partners.slurm

cd $PYTHONPATH/zsceval/scripts
python prep/select_hsp_seeds.py random3 -k 16          # inspect the seed list without training
python prep/gen_crossplay_yml.py random0 --heldout hsp --hsp_partners auto
```

Knobs are environment variables (`LAYOUTS SEEDS K STEPS EXP EXCLUDE CONCURRENCY
HSP_ROLLOUT_THREADS`) so they survive `sbatch --export`. `SEEDS` bypasses `select_hsp_seeds.py`
entirely, which is how a layout is topped up after losing seeds: selection is deterministic, so seed
*n* still denotes the same `w0` it did on the first attempt, and re-selecting with the same `-k`
would retrain everything that already succeeded. Budget is 2e6 steps against upstream's 1e7: a `random0` pilot at that
budget reached 156–200 sparse return against `select_bias_agent_br.py`'s filter threshold of 10.

Cross-play then takes `--heldout {sp,hsp,both}`, and `analyze_crossplay.py` reports the HSP partners
as their own group — pass `--partner_group heldout_hsp`. The two sets are deliberately *not* merged:
the ZSC metric averages over a single partner group, and mixing bias agents with self-play
checkpoints would average two different evaluations.

**Gotchas that cost real time:**

- **`.env` silently overrides pipeline defaults.** It exports a repo-wide `ROLLOUT_THREADS` (16 on
  the cluster, 4 locally) and is loaded *before* a script's own defaults, so `${ROLLOUT_THREADS:-12}`
  never applies. That variable is the PPO batch size, so each machine would train partners at a
  different one and the layouts stop being comparable. `hsp-partners.slurm` reads
  `HSP_ROLLOUT_THREADS` for exactly this reason. Any new pipeline wanting a value `.env` also sets
  needs its own name too.
- **`bigbatch` nodes have 14 CPUs, not 16.** Every `#SBATCH -c 16` in this repo was rejected with
  `cpu count per node cannot be satisfied`, which is why the slurm files sat unused while the runs
  happened on a workstation. `stampede` is the only 16-CPU partition and has a quarter of the memory.
  `hsp-partners.slurm` sizes its concurrency from `SLURM_CPUS_PER_TASK`, so `-c` can be overridden at
  submit time without editing the file; the other pipelines still hardcode their thread counts.
- **The experiment name is case-sensitive.** Upstream's `train_bias_agents.sh` writes `hsp-S1` while
  `extract_bias_agents_models.py` looks for `hsp-s1`, and the W&B filter does not fold case, so the
  upstream pair never matched. The fork defaults both to `hsp-s1`; the extractor takes an optional
  third positional to override it for pilots under another name.
- **The bias extractor writes `mid` and `final` only** — not `init`, unlike `extract_sp_models.py`.
  `HSP_TAGS` in `gen_crossplay_yml.py` defaults to `["final"]`.
- **The separated runner did not upload its checkpoints.** Bias agents train through
  `runner/separated/`, whose `save()` wrote each actor into `wandb.run.dir` but never called
  `wandb.save()`. Older wandb clients swept the run directory up at exit; 0.25.0 only uploads
  registered files, so every bias-agent checkpoint stayed local while the *shared* runner — which
  does call it — kept working. `extract_bias_agents_models.py` reads the W&B API, so it found an
  empty file list and died on `max() arg is an empty sequence` *after* logging the first run's
  return, which made it look like the second run was at fault. Both halves are fixed: the runner
  registers its checkpoints, and the extractor falls back to the run's own directory under
  `$PYTHONPATH/results/{env}/{layout}/{algo}/{exp}/wandb/run-*-{run_id}/files/` when W&B has none.
  Runs trained before the fix have checkpoints **only on the machine that trained them**.
- Extraction still reads the **W&B API** for run state and the `ep_sparse_r` curve that picks the
  `mid` checkpoint, so runs must be finished and synced. `WANDB_MODE=offline` means nothing is
  extracted until `util/sync-wandb.slurm` has run — worth checking before a job whose extraction
  step is 6 hours away.
- **The layouts do not train at one pace, and the wall clock was sized on the fastest.** Measured
  over array 46291 at concurrency 2: `random0` 3.2 h/seed (16 seeds in 26 h), `random3` 5.2 h/seed
  (~42 h), `unident_s` 5.9 h/seed (~47 h). The 36 h wall the pipeline used to carry cost the two
  slower layouts four seeds. It is now `72:00:00`, the cluster maximum; a `k` much above 16 needs
  higher concurrency or a split submission, because there is no more wall to buy.
- **A top-up submission is a one-task array, so its `%a` is 0 — the same as any other.** Job logs are
  `logs/hsp-partners-%A_%a.{log,err}` so two concurrent top-ups do not write the same file.
- **Concurrent seeds raced to create the shared run directory.** `train_bias_agent.py` used a
  test-then-`makedirs`, and the loser died before `wandb.init()` — leaving no W&B run at all, only a
  traceback in the per-seed log. Now `exist_ok=True`.
- A seed that dies does not abort its layout: extraction only picks up finished runs, so the
  survivors are still a usable pool. Per-seed output goes to `experiments/logs/hsp/{layout}/`
  (gitignored) — the batch loop only reports pass/fail, so that is where a traceback actually is.

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
