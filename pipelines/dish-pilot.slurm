#!/bin/bash -l
#SBATCH -J ZSCEvalDish
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/dish-pilot-%A_%a.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/dish-pilot-%A_%a.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 48:00:00

# Complex-dish pilot (Phase 2a, 15 Oct plan): can stage-1 self-play learn the
# `_cx` menus, where the best dish takes five ingredients of two kinds and an
# all-onion pot pays nothing? One array task per (layout, seed), stage 1 only,
# logged as `{arm}-cx` so nothing collides with the static-menu runs.
#
# Two arms, the hand-shaped baseline and MORL in its most competitive form
# (dense objectives annealed like the baseline's shaping, deliveries credited to
# the team), both on the `dishes` objective set so both log the same breakdown:
#   sbatch --array=0-5 dish-pilot.slurm                                   # bench_sp
#   ARM=bench_morl_ann MORL_TEAM_TASK=1 sbatch --array=0-5 --export=ALL dish-pilot.slurm
#
# Scripted references (what each cooking strategy is worth, a cook alone):
#   python experiments/dish_check.py   -> experiments/results/dish_check.json
# Read: ep_sparse_r against those, and ep_obj_dish_value / ep_obj_task_completion
# for whether the agents found the big mixed dishes or settled for small ones.

cd /home-mscluster/vnaidoo/Coding/masters-research/pipelines

if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
fi
if [ "$SHOULD_SOURCE_CONDA" = "true" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi
conda activate zsceval

# .env sets a repo-wide ROLLOUT_THREADS, and it is the PPO batch size; every
# stage-1 run we compare against used 12.
export ROLLOUT_THREADS=${DISH_ROLLOUT_THREADS:-12}
read -r -a LAYOUT_LIST <<< "${LAYOUTS:-unident_s_cx random1_cx}"
read -r -a SEED_LIST <<< "${SEEDS:-1 2 3}"
ARM=${ARM:-bench_sp}
export OBJECTIVES=${OBJECTIVES:-dishes}
# Five-ingredient dishes are a longer credit chain than the static menu's; the
# schedules scale with the budget (train_morl_benchmark.sh).
export MORL_BENCH_STEPS=${MORL_BENCH_STEPS:-5e6}

IDX=${SLURM_ARRAY_TASK_ID:-0}
N=$(( ${#LAYOUT_LIST[@]} * ${#SEED_LIST[@]} ))
if [ "$IDX" -ge "$N" ]; then
    echo "array index ${IDX} is past the ${N} (layout, seed) pairs; nothing to do"
    exit 0
fi
LAYOUT=${LAYOUT_LIST[$(( IDX / ${#SEED_LIST[@]} ))]}
SEED=${SEED_LIST[$(( IDX % ${#SEED_LIST[@]} ))]}
echo "dish pilot: layout ${LAYOUT}, arm ${ARM}, seed ${SEED}, objectives ${OBJECTIVES}," \
     "steps ${MORL_BENCH_STEPS}, rollout_threads ${ROLLOUT_THREADS}, team_task ${MORL_TEAM_TASK:-0}"

# Seeds of one experiment share a results directory; stagger the starts.
sleep $(( IDX * 30 ))
cd $PYTHONPATH/zsceval/scripts/overcooked
python morl/check_complex_dishes.py || exit 1
EXTRA_FLAGS="${DISH_FLAGS:-}" EXP_SUFFIX=${EXP_SUFFIX:--cx} \
    bash shell/train_morl_benchmark.sh $LAYOUT $ARM $SEED $SEED
