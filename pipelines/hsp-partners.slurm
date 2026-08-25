#!/bin/bash -l
#SBATCH -J ZSCEvalHSPPartners
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/hsp-partners-%a.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/hsp-partners-%a.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 24:00:00
#SBATCH --array=0-2

# Train the HSP bias agents that ZSC-Eval evaluates zero-shot coordination
# against, so the ZSC metric stops depending on held-out self-play checkpoints.
#
# Why this exists: gen_crossplay_yml.py drew held-out partners from a pre-existing
# `sp` pool that only ever existed for random0, which made a cross-layout
# comparison impossible. HSP bias agents can be produced for any layout, and are
# what the benchmark actually intends -- self-play checkpoints are far less
# behaviourally diverse than reward-randomised agents.
#
# One array task per layout. The three are independent -- separate seeds,
# separate W&B runs, separate pool directories -- so they are submitted as an
# array rather than a loop: the cluster runs them concurrently, and a layout
# that fails does not take the other two down with it.
#
#   task 0  random0     forced coordination, 30-candidate w0 space
#   task 1  random3     52-candidate space
#   task 2  unident_s   52-candidate space
#
# Budget: upstream trains these at 1e7 steps. A pilot on random0 at 2e6 (entropy
# horizons scaled to the budget) reached 156-200 sparse return against a filter
# threshold of 10, so a fifth of the upstream budget is enough here. Of six
# pilot agents, five cleared the filter; the one that did not is idle-dominant,
# which prep/select_hsp_seeds.py now excludes up front. At 2 concurrent runs a
# 2e6 seed takes ~40 min, so 16 seeds is ~5-6 h per layout.
#
# Seed choice is delegated to prep/select_hsp_seeds.py rather than a range: w0 is
# an *enumeration*, not a random draw, so a contiguous or evenly-strided range
# aliases onto a corner of the space. See that script's docstring. Selection is
# deterministic, so the seed list here is reproducible from the layout alone.
#
# Usage:
#   sbatch hsp-partners.slurm                       # all three layouts
#   sbatch --array=0 hsp-partners.slurm             # random0 only
#   sbatch --export=ALL,K=8,STEPS=1e6 hsp-partners.slurm
#   bash hsp-partners.slurm random3                 # locally, one layout
#
# Knobs (environment, so they survive --export): LAYOUTS K STEPS EXP EXCLUDE
# CONCURRENCY ROLLOUT_THREADS DUMMY_BATCH_SIZE TRAINING_THREADS.

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [ -f "$SUBMIT_DIR/../.env" ]; then
    export $(grep -v '^#' "$SUBMIT_DIR/../.env" | xargs)
    echo "Environment variables loaded from .env"
fi

if [ "$SHOULD_SOURCE_CONDA" = "true" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    echo "Skipping Conda source."
fi

conda activate zsceval

# shellcheck disable=SC2206
LAYOUT_LIST=(${LAYOUTS:-random0 random3 unident_s})
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    LAYOUT=${LAYOUT_LIST[$SLURM_ARRAY_TASK_ID]}
    if [ -z "$LAYOUT" ]; then
        echo "array index $SLURM_ARRAY_TASK_ID past the end of (${LAYOUT_LIST[*]})"
        exit 1
    fi
else
    LAYOUT=${1:-${LAYOUT_LIST[0]}}
fi

K=${K:-16}
STEPS=${STEPS:-2e6}
EXP=${EXP:-hsp-s1}
EXCLUDE=${EXCLUDE:-}

# Measured on a 20-thread box: throughput peaks near 5 worker processes, and two
# concurrent runs give ~2x aggregate while three collapse. ROLLOUT_THREADS is
# split into worker processes by DUMMY_BATCH_SIZE (12/4 = 3 workers + a main),
# so CONCURRENCY=2 fits the 16 cores this job asks for.
export ROLLOUT_THREADS=${ROLLOUT_THREADS:-12}
export DUMMY_BATCH_SIZE=${DUMMY_BATCH_SIZE:-4}
export TRAINING_THREADS=${TRAINING_THREADS:-4}
export HSP_EVAL_THREADS=${HSP_EVAL_THREADS:-20}
CONCURRENCY=${CONCURRENCY:-2}

# Per-seed training output. The batch loop only reports pass/fail, so this is
# where a failed seed's traceback actually is. experiments/logs/ is gitignored.
LOGDIR="$SUBMIT_DIR/../experiments/logs/hsp/$LAYOUT"
mkdir -p "$LOGDIR"

cd $PYTHONPATH/zsceval/scripts || exit 1

EXCLUDE_FLAG=""
[ -n "$EXCLUDE" ] && EXCLUDE_FLAG="--exclude $EXCLUDE"
# select_hsp_seeds.py logs the candidate count and how many idle-dominant
# vectors it dropped to stderr; that is the provenance for the seed list, so it
# is kept in the job log rather than discarded, and reprinted whole on failure.
SEED_OUT=$(python prep/select_hsp_seeds.py $LAYOUT -k $K $EXCLUDE_FLAG 2>&1)
SEEDS=$(printf '%s\n' "$SEED_OUT" | grep '^seeds: ' | sed 's/^seeds: //')
if [ -z "$SEEDS" ]; then
    echo "seed selection failed for $LAYOUT:"
    printf '%s\n' "$SEED_OUT"
    exit 1
fi
printf '%s\n' "$SEED_OUT" | grep -E "candidates,|usable candidates|dropped"

echo "=========================================================="
echo "layout      $LAYOUT"
echo "exp         $EXP"
echo "steps       $STEPS"
echo "seeds (k=$K) $SEEDS"
echo "concurrency $CONCURRENCY"
echo "logs        $LOGDIR"
echo "started     $(date)"
echo "=========================================================="

cd overcooked || exit 1

FAILED=""

# A seed that dies must not abort the layout: extraction only ever picks up
# runs that finished and synced, so the survivors are still a usable pool. The
# failures are collected and reported at the end instead.
run_batch () {
    local seeds=("$@")
    local pids=()
    local s i
    for s in "${seeds[@]}"; do
        bash shell/train_bias_agents.sh "$LAYOUT" "$s" "$STEPS" "$EXP" \
            > "$LOGDIR/seed$s.log" 2>&1 &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "  seed ${seeds[$i]} ok"
        else
            echo "  seed ${seeds[$i]} FAILED -- see $LOGDIR/seed${seeds[$i]}.log"
            FAILED="$FAILED ${seeds[$i]}"
        fi
    done
}

BATCH=()
for SEED in $SEEDS; do
    BATCH+=("$SEED")
    if [ ${#BATCH[@]} -eq "$CONCURRENCY" ]; then
        echo "########## $LAYOUT: training ${BATCH[*]} ##########"
        run_batch "${BATCH[@]}"
        echo "batch done at $(date +%H:%M:%S)"
        BATCH=()
    fi
done
if [ ${#BATCH[@]} -gt 0 ]; then
    echo "########## $LAYOUT: training ${BATCH[*]} ##########"
    run_batch "${BATCH[@]}"
    echo "batch done at $(date +%H:%M:%S)"
fi

if [ -n "$FAILED" ]; then
    echo "WARNING: seeds failed to train:$FAILED"
fi

# Extraction reads the W&B API, so the runs above must be finished and synced.
# In offline mode nothing is extracted until `wandb sync` has run -- see
# util/sync-wandb.slurm.
#
# The third positional overrides the built-in experiment-name map, which is
# needed for any exp other than the default.
cd .. || exit 1
python extract_models/extract_bias_agents_models.py $LAYOUT overcooked $EXP || exit 1

echo "=========================================================="
echo "HSP PARTNERS DONE for $LAYOUT at $(date)"
echo "Pool: $POLICY_POOL/$LAYOUT/hsp/s1/$EXP"
echo "Next: python prep/gen_crossplay_yml.py $LAYOUT --heldout hsp --hsp_partners auto --hsp_exp $EXP"
echo "=========================================================="
