#!/bin/bash
# Re-baseline the MORL arms under the anchored coordination objective.
#
# The three arms whose reward is w . r_vec were all trained against a farmable
# objective. On random0 `coordination` counts counter handoffs, the put and the
# pickup undo each other, and the measured ratio of the best score reached while
# delivering nothing to the median reached while delivering is 3.69 -- the
# objective was better farmed than earned. Six stage-1 seeds across two arms
# found the loop. Those runs measure reward hacking, not MORL.
#
# Not re-run, because neither uses the objective vector as its reward:
#   bench_sp      sparse + hand-shaped
#   bench_sparse  w = (20,0,0,0), task_completion only, which is not farmable
# The baseline is therefore intact and only the treatment arms move.
#
# The gate for this was a single pilot: bench_morl_div seed 6 (plating +
# coordination) scored sparse 0 with coordination 317.8 under `default`, and
# sparse 196.7 with coordination 84.2 under `anchored`, same seed, same weights,
# same budget. Anchoring alone was sufficient -- no diminishing alpha and no
# task rescale -- so those levers stay in reserve rather than being load-bearing.
#
# random0 only. On unident_s `coordination` is identically 0 because the layout
# has no counter handoffs, so anchoring changes nothing there; what that layout
# needs is renormalising w over the objectives that actually fire, which is a
# separate fix.
#
# Runs sequentially on one machine and is expected to take hours. Per-run logs
# go to experiments/logs/rebaseline/ (gitignored).
#
# Usage: bash rebaseline-anchored.sh [layout] [arms] [seed_begin] [seed_max]

set -u
ROOT=/home/elementrix/coding/masters-research/.claude/worktrees/partner-effects
LAYOUT=${1:-random0}
ARMS=${2:-"bench_morl bench_morl_ad bench_morl_div"}
SEED_BEGIN=${3:-1}
SEED_MAX=${4:-6}

export $(grep -v '^#' "$ROOT/.env" | xargs)
export PYTHONPATH="$ROOT/zsc-eval"
export POLICY_POOL="$ROOT/zsc-eval/zsceval/scripts/overcooked/policy_pool"
# The arms being replaced trained at 12; .env sets 4 here and rollout threads are
# the PPO batch size, so the default would make old and new incomparable.
export ROLLOUT_THREADS=${REBASELINE_ROLLOUT_THREADS:-12}
export TRAINING_THREADS=2
export OBJECTIVES=anchored
# Keeps these runs from sharing an experiment_name with the ones they supersede.
export EXP_SUFFIX=${EXP_SUFFIX:--anc}
# train_morl_benchmark.sh invokes bare `python`.
export PATH=/home/elementrix/miniconda3/envs/zsceval/bin:$PATH

LOGDIR="$ROOT/experiments/logs/rebaseline"
mkdir -p "$LOGDIR"
cd "$PYTHONPATH/zsceval/scripts/overcooked"

echo "re-baseline: layout ${LAYOUT}, arms [${ARMS}], seeds ${SEED_BEGIN}..${SEED_MAX}"
echo "objectives=${OBJECTIVES}, experiment suffix=${EXP_SUFFIX}, threads=${ROLLOUT_THREADS}"

for ARM in $ARMS; do
    for SEED in $(seq "$SEED_BEGIN" "$SEED_MAX"); do
        LOG="${LOGDIR}/${LAYOUT}_${ARM}${EXP_SUFFIX}_s${SEED}.log"
        echo "=== ${ARM} seed ${SEED} -> ${LOG}"
        bash shell/train_morl_benchmark.sh "$LAYOUT" "$ARM" "$SEED" "$SEED" \
            > "$LOG" 2>&1 || echo "  FAILED ${ARM} seed ${SEED}"
    done
done

echo "=========================================================="
echo "re-baseline done for ${LAYOUT} at $(date)"
echo "logs in ${LOGDIR}"
echo "=========================================================="
