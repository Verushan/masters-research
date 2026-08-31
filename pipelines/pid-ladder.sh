#!/bin/bash
# Partner-conditioning ablation: does a stage-2 agent coordinate better when it
# is told *which* partner it is playing?
#
# THIS IS AN ORACLE CEILING, NOT A ZSC METHOD. A held-out partner has no id the
# agent was ever trained on, so nothing here can generalise zero-shot by
# construction. Its value is that the gap between a rung and rung1 separates
# partner *uncertainty* (closed by knowing who you are with) from partner
# *diversity* (coordinating badly even when you know). See CLAUDE.md.
#
# The rungs differ only in what the ACTOR sees; every one of them gets the
# upstream critic-side --use_agent_policy_id, so the population, the reward, the
# budget and the schedules are identical across the ladder.
#
#   rung1  control      -- critic sees the partner id, actor does not
#   rung2  raw scalar   -- actor also sees id = (i+1)/len(population), the same
#                          scalar the critic gets. Pool-size dependent, and it
#                          imposes an ordinal structure on what is categorical:
#                          partners adjacent in the yml get adjacent ids.
#   rung3  one-hot      -- actor sees the id one-hot over the population, which
#                          fixes both of rung2's problems.
#
# Usage: bash pid-ladder.sh [layout] [arm] [seed_begin] [seed_max] [rungs...]
#   bash pid-ladder.sh random0 bench_sp 2 3 rung1 rung3
#
# Runs the rungs CONCURRENTLY (one process each) and waits for all of them, so
# the wall clock is one rung's, not the ladder's. Per-rung output goes to
# experiments/logs/pid-ladder/ -- the loop below only reports pass/fail.

cd "$(dirname "$0")" || exit 1

if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
    echo "Environment variables loaded from .env"
fi

if [ "$SHOULD_SOURCE_CONDA" = "true" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate zsceval
fi

LAYOUT=${1:-random0}
ARM=${2:-bench_sp}
SEED_BEGIN=${3:-1}
SEED_MAX=${4:-3}
# Guarded: a bare `shift 4` with fewer than four arguments fails and leaves
# "$@" holding the layout, which would then be read as a rung name.
if [ $# -gt 4 ]; then
    shift 4
    RUNGS="$*"
else
    RUNGS="rung1 rung3"
fi
STEPS=${STEPS:-2e6}

# The trap that .env sets and every pipeline has had to re-pin: ROLLOUT_THREADS
# is loaded above, so it takes whichever value the machine happens to carry (4 on
# the workstation, 16 on the cluster). That variable is the PPO batch size, and
# the ladder is a *within-experiment comparison* -- a rung trained at a different
# batch size than the rung it is compared against is not a partner-conditioning
# result, it is a batch-size result. The seed-1 runs were trained at 12, so 12 is
# what the rest of the ladder has to use.
export ROLLOUT_THREADS=${LADDER_ROLLOUT_THREADS:-12}

LOGS=$(cd .. && pwd)/experiments/logs/pid-ladder/$LAYOUT
mkdir -p "$LOGS"

cd "$PYTHONPATH/zsceval/scripts/overcooked" || exit 1

yml="${POLICY_POOL}/${LAYOUT}/fcp/s2/train-${ARM}.yml"
if [[ ! -f "${yml}" ]]; then
    echo "missing ${yml} -- run prep/gen_arm_S2_yml.py ${LAYOUT} --arm ${ARM} first"
    exit 1
fi

echo "=========================================================="
echo "layout      $LAYOUT"
echo "arm         $ARM"
echo "rungs       $RUNGS"
echo "seeds       $SEED_BEGIN..$SEED_MAX"
echo "steps       $STEPS"
echo "threads     $ROLLOUT_THREADS rollout / $TRAINING_THREADS training"
echo "logs        $LOGS"
echo "started     $(date)"
echo "=========================================================="

declare -A PIDS
for rung in $RUNGS; do
    case "$rung" in
        # PID_OBS unset entirely: an empty string is what the shell script tests
        # for, so exporting PID_OBS="" would be the control by accident anyway --
        # but unsetting it says so.
        rung1) unset PID_OBS; unset PID_OBS_DIM ;;
        # dim 0 is the raw scalar the critic already uses, not "no id".
        rung2) export PID_OBS=1 PID_OBS_DIM=0 ;;
        # Default width == the yml's entry count, which is what
        # policy_pool.load_population divides by. Left to the shell script so
        # there is one place that knows it.
        rung3) export PID_OBS=1; unset PID_OBS_DIM ;;
        *) echo "unknown rung '$rung'"; exit 1 ;;
    esac
    log="$LOGS/${ARM}-${rung}-s${SEED_BEGIN}_${SEED_MAX}.log"
    echo "launching $rung -> $log"
    bash shell/train_morl_stage_2.sh "$LAYOUT" "$ARM" "$SEED_BEGIN" "$SEED_MAX" "$STEPS" "-$rung" \
        > "$log" 2>&1 &
    PIDS[$rung]=$!
done

fail=0
for rung in "${!PIDS[@]}"; do
    if wait "${PIDS[$rung]}"; then
        echo "  $rung ok"
    else
        echo "  $rung FAILED -- see $LOGS/${ARM}-${rung}-s${SEED_BEGIN}_${SEED_MAX}.log"
        fail=1
    fi
done

echo "=========================================================="
echo "PID LADDER DONE for $LAYOUT/$ARM at $(date)"
echo "Next: python extract_models/extract_S2_models.py --layout $LAYOUT --env overcooked \\"
echo "        --algorithm fcp --exp fcp-S2-${ARM}-{rung}"
echo "=========================================================="
exit $fail
