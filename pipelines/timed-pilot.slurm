#!/bin/bash -l
#SBATCH -J ZSCEvalTimed
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/timed-pilot-%A_%a.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/timed-pilot-%A_%a.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 48:00:00

# Timed-orders pilot: can ordinary self-play learn to work the order queue at
# the usual budget? Stage 1 only, one array task per (layout, seed), arm
# bench_sp (the hand-shaped baseline reward) with --timed_orders, logged as
# experiment bench_sp-timed so nothing collides with the static-menu runs.
#
# Training only. morl-benchmark.slurm's extraction step names experiments by
# arm without the suffix, so it would overwrite the existing bench_sp pool;
# extract these explicitly once they are judged worth keeping:
#   python extract_models/extract_sp_models.py unident_s_m overcooked bench_sp-timed
#
#   sbatch --array=0-5 timed-pilot.slurm                      # 2 layouts x seeds 1-3
#   LAYOUTS="unident_s_m" SEEDS="4 5" sbatch --array=0-1 --export=ALL timed-pilot.slurm
#
# Read the curves: ep_sparse_r (pay - penalties), ep_on_time_rate, ep_orders_expired,
# against the scripted references in experiments/results/timed_orders_check.json.

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
export ROLLOUT_THREADS=${TIMED_ROLLOUT_THREADS:-12}
read -r -a LAYOUT_LIST <<< "${LAYOUTS:-unident_s_m random1_m}"
read -r -a SEED_LIST <<< "${SEEDS:-1 2 3}"
ARM=${ARM:-bench_sp}

IDX=${SLURM_ARRAY_TASK_ID:-0}
N=$(( ${#LAYOUT_LIST[@]} * ${#SEED_LIST[@]} ))
if [ "$IDX" -ge "$N" ]; then
    echo "array index ${IDX} is past the ${N} (layout, seed) pairs; nothing to do"
    exit 0
fi
LAYOUT=${LAYOUT_LIST[$(( IDX / ${#SEED_LIST[@]} ))]}
SEED=${SEED_LIST[$(( IDX % ${#SEED_LIST[@]} ))]}
echo "timed pilot: layout ${LAYOUT}, arm ${ARM}, seed ${SEED}, rollout_threads ${ROLLOUT_THREADS}"

cd $PYTHONPATH/zsceval/scripts/overcooked
python morl/check_timed_orders.py || exit 1
EXTRA_FLAGS="--timed_orders ${TIMED_FLAGS:-}" EXP_SUFFIX=${EXP_SUFFIX:--timed} \
    bash shell/train_morl_benchmark.sh $LAYOUT $ARM $SEED $SEED
