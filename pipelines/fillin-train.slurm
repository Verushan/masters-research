#!/bin/bash -l
#SBATCH -J ZSCEvalFillIn
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/fillin-train-%A_%a.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/fillin-train-%A_%a.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 48:00:00

# Fill-in suite, Step 7: train the matched arms against scripted specialists.
#
# Every arm is a stage-2 run (recurrent ego, both seats, every partner) against
# the same scripted population -- potter, server, idle, dial 2, dial 8 -- and the
# same mid-episode swaps: at step 200, with probability 0.5, the partner is
# replaced by a different script from that set. Dial 5, the generalist, the
# clutter script and the 133/266 swap timings are held out for Step 8.
#
# The arms differ only in the ego's reward:
#
#   hand     the hand-shaped reward (sparse + annealed shaping). `tasks` is passed
#            only so the task breakdown is logged; it does not touch the reward.
#   neglect  the task reward, each dense task weighted by 1 - the partner's
#            recent share of it, weights shown to the agent, dense tasks annealed
#            with the shaping horizon, deliveries credited to the team.
#   noann    neglect without the anneal: the dense task terms stay on for the
#            whole run. Isolates whether annealing throws the mechanism away.
#   neglect2 / noann2   the same two with the corrected weight rule (post hoc,
#            exploratory). Step 8 found the original rule never relaxes: after
#            a swap the agent keeps being paid nothing for any task the old
#            partner did. A prior pseudo-count (--morl_neglect_prior 0.5) and a
#            10-step half-life let the weights return to even within ~20 steps.
#            Run with SPEC="neglect2:1-5 neglect2:6-10 noann2:1-5 noann2:6-10".
#   handns   hand without the swaps: the same five specialists, one per game.
#            Arm `hand` changed two things at once -- who it trains with, and
#            swaps during training -- and this separates them.
#            Run with SPEC="handns:1-5 handns:6-10" ARMS="handns".
#
# There is no fixed-weight task arm: `tasks` at 20,3,3,5 with team credit and
# annealing is the hand-shaped reward exactly, step for step
# (morl/check_fillin_training.py), so it would be a second copy of `hand`.
#
# SPEC is arm:seed_begin-seed_max, one per array index. The default is the three
# arms x 10 seeds as six 5-seed tasks, which fits the 6-running-job QOS.
#
# Usage: sbatch --array=0-5 fillin-train.slurm
#        SPEC="neglect:11-12" sbatch --array=0 --export=ALL fillin-train.slurm
#        MODE=extract sbatch fillin-train.slurm        # after all tasks finish
#
# EXP_TAG is appended to the experiment name (fcp-S2-scripted-{arm}{EXP_TAG}),
# so a run at a different budget cannot share a name -- and so a seed number --
# with the 2e6 runs: extract_S2_models.py writes {seed}.pt and raises on
# duplicate seeds. Pass the same EXP_TAG to MODE=extract.
#        LAYOUT=random3 STEPS=1e7 EXP_TAG=-1e7 SPEC="hand:1-1 hand:2-2" ARMS=hand \
#            sbatch --array=0-1 --export=ALL fillin-train.slurm

cd /home-mscluster/vnaidoo/Coding/masters-research/pipelines

if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
    echo "Environment variables loaded from .env"
fi

if [ "$SHOULD_SOURCE_CONDA" = "true" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    echo "Skipping Conda source."
fi

conda activate zsceval

# Same trap as every other pipeline: .env sets a repo-wide ROLLOUT_THREADS, and
# it is the PPO batch size. Every stage-2 run we compare against used 12.
export ROLLOUT_THREADS=${FILLIN_ROLLOUT_THREADS:-12}

LAYOUT=${LAYOUT:-unident_s}
STEPS=${STEPS:-2e6}
POP=scripted
SPEC=${SPEC:-"hand:1-5 hand:6-10 neglect:1-5 neglect:6-10 noann:1-5 noann:6-10"}
ARMS=${ARMS:-"hand neglect noann"}
EXP_TAG=${EXP_TAG:-""}

cd $PYTHONPATH/zsceval/scripts

if [ "${MODE}" = "extract" ]; then
    EXP_FLAGS=""
    for ARM in $ARMS; do
        EXP_FLAGS="$EXP_FLAGS --exp fcp-S2-${POP}-${ARM}${EXP_TAG}"
    done
    # The final checkpoint, as fillin-crossplay.slurm takes it, so the fill-in
    # suite and the zero-shot matrix score the same file. The task-weights
    # config is what both load the neglect arms through.
    python extract_models/extract_S2_models.py --layout $LAYOUT --env Overcooked -a fcp $EXP_FLAGS \
        --checkpoint final || exit 1
    python prep/store_policy_config.py $LAYOUT --morl_obs_weights tasks || exit 1
    for ARM in $ARMS; do
        echo "fcp-S2-${POP}-${ARM}${EXP_TAG}: $(ls $POLICY_POOL/$LAYOUT/fcp/s2/fcp-S2-${POP}-${ARM}${EXP_TAG}/ 2>/dev/null | wc -l) checkpoints"
    done
    exit 0
fi

read -r -a TASKS <<< "$SPEC"
IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "${#TASKS[@]}" ]; then
    echo "array index ${IDX} is past the ${#TASKS[@]}-entry spec; nothing to do"
    exit 0
fi
TASK=${TASKS[$IDX]}
ARM=${TASK%%:*}
SEEDS=${TASK##*:}
SEED_BEGIN=${SEEDS%%-*}
SEED_MAX=${SEEDS##*-}

# The swap pool is the training partners, by script name, as the yml generator
# reports them -- so the two cannot drift apart.
SWAP_POOL=$(python prep/gen_script_population_yml.py $LAYOUT --name $POP | sed -n 's/^swap pool: //p')
if [ -z "$SWAP_POOL" ]; then
    echo "gen_script_population_yml.py did not report a swap pool"
    exit 1
fi
SWAPS="--script_swap_steps 200 --script_swap_pool ${SWAP_POOL} --script_swap_prob 0.5"

NEGLECT="--use_morl --morl_objectives tasks --morl_weights 20,3,3,5 --morl_team_task \
--morl_adaptive_target neglect --use_morl_obs_weights"
RELAX="--morl_neglect_prior 0.5 --morl_neglect_halflife 10"
case $ARM in
    hand)    REWARD="--morl_objectives tasks" ;;
    handns)  REWARD="--morl_objectives tasks"; SWAPS="" ;;
    neglect) REWARD="${NEGLECT} --morl_anneal_dense" ;;
    noann)   REWARD="${NEGLECT}" ;;
    neglect2) REWARD="${NEGLECT} ${RELAX} --morl_anneal_dense" ;;
    noann2)   REWARD="${NEGLECT} ${RELAX}" ;;
    *) echo "unknown arm '${ARM}' (expected one of: ${ARMS})"; exit 1 ;;
esac
export EGO_FLAGS="${REWARD} ${SWAPS}"

# Stagger the array: two tasks of one arm share a run directory, and starting
# together they raced first on creating it (fixed with exist_ok) and then inside
# wandb.init on its wandb/ subdirectory (job 60603, task 1: FileNotFoundError).
sleep $((IDX * 60))

echo "fill-in task ${IDX}: layout ${LAYOUT}, arm ${ARM}, seeds ${SEED_BEGIN}-${SEED_MAX}, steps ${STEPS}, exp fcp-S2-${POP}-${ARM}${EXP_TAG}"
echo "rollout_threads=${ROLLOUT_THREADS}"

cd overcooked
bash shell/train_morl_stage_2.sh $LAYOUT $POP $SEED_BEGIN $SEED_MAX $STEPS "-${ARM}${EXP_TAG}"
