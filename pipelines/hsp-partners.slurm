#!/bin/bash -l
#SBATCH -J ZSCEvalHSPPartners
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/hsp-partners-%A_%a.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/hsp-partners-%A_%a.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 72:00:00
#SBATCH --array=0-2

# %A_%a, not %a: a top-up submission is a one-task array, so two of them
# concurrently are both task 0 and would write the same log. The job id keeps
# them apart and keeps the previous attempt's log around to compare against.

# -c 14 is one whole bigbatch node (14 CPUs, 124 GB). The partition rejects
# anything larger -- that is what "cpu count per node cannot be satisfied"
# means, and every pipeline in this repo asked for 16 until it was measured.
# stampede has 16 CPUs but a quarter of the memory; bigbatch's 48 nodes make a
# 3-task array easy to place. The concurrency below sizes itself to whatever
# the allocation actually is, so changing -c needs no other edit.

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
# which prep/select_hsp_seeds.py now excludes up front.
#
# -t is 72:00:00, the cluster maximum, because the layouts do not run at one
# pace. Measured over array 46291 at concurrency 2: random0 3.2 h/seed (16 seeds
# in 26 h), random3 5.2 h/seed (~42 h), unident_s 5.9 h/seed (~47 h). The 36 h
# wall this file used to carry was sized on random0 alone and cost the other two
# layouts four seeds between them. 47 h fits in 72 h; nothing larger does, so a
# k much above 16 needs either higher concurrency or a split submission.
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
#   # top up one layout with named seeds, skipping selection entirely
#   LAYOUTS=random3 SEEDS="41 42 50 51" sbatch --array=0 --export=ALL hsp-partners.slurm
#
# Knobs (environment, so they survive --export): LAYOUTS SEEDS K STEPS EXP
# EXCLUDE CONCURRENCY HSP_ROLLOUT_THREADS DUMMY_BATCH_SIZE TRAINING_THREADS.

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
# concurrent runs give ~2x aggregate while three collapse.
#
# ROLLOUT_THREADS is deliberately NOT scaled to the allocation: it is the number
# of envs in the PPO batch, so changing it changes the learning dynamics, and a
# partner set trained at two different batch sizes is not one partner set.
# Concurrency is pure scheduling and can safely follow the node.
#
# The name is deliberately not ROLLOUT_THREADS: .env exports a repo-wide value
# (16 on the cluster, 4 locally), so a ${ROLLOUT_THREADS:-12} default would
# never have applied and the partner sets would have been trained at whatever
# batch size the machine's .env happened to carry.
export ROLLOUT_THREADS=${HSP_ROLLOUT_THREADS:-12}
export DUMMY_BATCH_SIZE=${DUMMY_BATCH_SIZE:-4}
export TRAINING_THREADS=${TRAINING_THREADS:-4}
export HSP_EVAL_THREADS=${HSP_EVAL_THREADS:-20}

# Fit the batch size to the allocation instead of assuming the 16 cores this
# file used to ask for. ROLLOUT_THREADS envs are split into worker processes of
# DUMMY_BATCH_SIZE each, plus a main process per run. Oversubscribing is worse
# than running fewer seeds at once: the workers step in lockstep, so a run
# proceeds at the pace of its slowest env.
CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
PER_RUN=$(( ROLLOUT_THREADS / DUMMY_BATCH_SIZE + 1 ))
if [ -z "$CONCURRENCY" ]; then
    CONCURRENCY=$(( CPUS / PER_RUN ))
    [ "$CONCURRENCY" -lt 1 ] && CONCURRENCY=1
    [ "$CONCURRENCY" -gt 2 ] && CONCURRENCY=2
fi

# Per-seed training output. The batch loop only reports pass/fail, so this is
# where a failed seed's traceback actually is. experiments/logs/ is gitignored.
LOGDIR="$SUBMIT_DIR/../experiments/logs/hsp/$LAYOUT"
mkdir -p "$LOGDIR"

cd $PYTHONPATH/zsceval/scripts || exit 1

# An explicit SEEDS list bypasses selection. This is how a layout is topped up
# after losing seeds to the wall clock: selection is deterministic, so seed n
# still denotes the same w0 it did on the first attempt, and re-selecting with
# the same -k would retrain every seed that already succeeded.
if [ -n "$SEEDS" ]; then
    # commas are accepted because sbatch --export splits its own value on them;
    # the reliable way to pass a space-separated list is to export it first.
    SEEDS=$(echo "$SEEDS" | tr ',' ' ')
    SEED_SOURCE="explicit"
else
    SEED_SOURCE="select_hsp_seeds.py (k=$K)"
    EXCLUDE_FLAG=""
    [ -n "$EXCLUDE" ] && EXCLUDE_FLAG="--exclude $EXCLUDE"
    # select_hsp_seeds.py logs the candidate count and how many idle-dominant
    # vectors it dropped to stderr; that is the provenance for the seed list, so
    # it is kept in the job log rather than discarded, and reprinted on failure.
    SEED_OUT=$(python prep/select_hsp_seeds.py $LAYOUT -k $K $EXCLUDE_FLAG 2>&1)
    SEEDS=$(printf '%s\n' "$SEED_OUT" | grep '^seeds: ' | sed 's/^seeds: //')
    if [ -z "$SEEDS" ]; then
        echo "seed selection failed for $LAYOUT:"
        printf '%s\n' "$SEED_OUT"
        exit 1
    fi
    printf '%s\n' "$SEED_OUT" | grep -E "candidates,|usable candidates|dropped"
fi

echo "=========================================================="
echo "layout      $LAYOUT"
echo "exp         $EXP"
echo "steps       $STEPS"
echo "seeds       $SEEDS"
echo "  from      $SEED_SOURCE"
echo "cpus        $CPUS ($PER_RUN per run)"
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
