#!/bin/bash -l
#SBATCH -J ZSCEvalMORLS2
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/morl-s2-%j.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/morl-s2-%j.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 48:00:00

# MORL benchmark, stage 2: put each arm's population through the rest of the
# ZSC-Eval pipeline.
#
# The stage-1 benchmark (morl-benchmark.slurm) measured what each reward
# produces in self-play, and found that MORL agents coordinate better with
# themselves and much worse with strangers. That is precisely the failure
# population-based training is supposed to repair, so it leaves the question
# the proposal actually cares about unanswered: what does FCP produce when the
# population it trains against was raised on the objective vector?
#
# This runs stage 2 once per arm. The population is that arm's three stage-1
# seeds at init/mid/final -- 9 partners, the same count for every arm, so the
# only thing separating the runs is still the reward those partners optimised.
# The ego agent's own reward is the task reward in all arms (see the header of
# shell/train_morl_stage_2.sh for why).
#
# Budget: num_env_steps is per run and is NOT multiplied by population size in
# stage 2 (the `*= population_size` in overcooked_runner.train_mep is the
# stage-1 branch), so 2e6 here is 2e6 -- matched to stage 1, and 4x the 5e5 the
# existing fcp-S2-s16 agent got, which the stage-1 report flagged as
# undertrained. ARMS x SEEDS runs at ~2-3h each; trim either to fit a shorter
# queue and run the rest as a second job.
#
# The trailing suffix lands in the experiment name (fcp-S2-{arm}{sfx}) and the
# pool directory, so a reduced pilot ("-pilot") and the full run can coexist
# instead of the second silently overwriting the first, which is what would
# happen otherwise -- extract_S2_models.py keys checkpoints by seed alone.
#
# Usage: sbatch morl-s2.slurm  [layout] [arms] [seed_begin] [seed_max] [steps] [exp_suffix]

if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
    echo "Environment variables loaded from wandb.env"
fi

if [ "$SHOULD_SOURCE_CONDA" = "true" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    echo "Skipping Conda source."
fi

conda activate zsceval

# The trap MORL_ROLLOUT_THREADS and HSP_ROLLOUT_THREADS both exist for: .env
# exports a repo-wide ROLLOUT_THREADS (16 on the cluster, 4 on the workstation)
# and is loaded above, so this would otherwise take whichever value the machine
# happens to carry. It is the PPO batch size. Every stage-1 arm these
# populations are built from was trained at 12, and so were the pid-ladder
# stage-2 runs -- a stage-2 arm at 16 is not comparable with either.
export ROLLOUT_THREADS=${S2_ROLLOUT_THREADS:-12}

LAYOUT=${1:-random0}
# bench_sparse is left out on purpose: its stage-1 agents never learned to
# deliver, so a population of them is a population of near-random partners and
# the stage-2 run measures nothing about the reward.
ARMS=${2:-"bench_sp bench_morl bench_morl_ad"}
SEED_BEGIN=${3:-1}
SEED_MAX=${4:-3}
NUM_ENV_STEPS=${5:-2e6}
EXP_SUFFIX=${6:-""}

cd $PYTHONPATH/zsceval/scripts

# Fails loudly on a missing stage-1 checkpoint rather than quietly training
# against a smaller population, which would break the size matching.
ARM_FLAGS=""
for ARM in $ARMS; do
    ARM_FLAGS="$ARM_FLAGS --arm $ARM"
done
python prep/gen_arm_S2_yml.py $LAYOUT $ARM_FLAGS || exit 1

cd overcooked
for ARM in $ARMS; do
    bash shell/train_morl_stage_2.sh $LAYOUT $ARM $SEED_BEGIN $SEED_MAX $NUM_ENV_STEPS "$EXP_SUFFIX" || exit 1
done

# Extraction reads the W&B API, so the runs above must be finished and synced.
# Each arm's stage-2 experiment is named fcp-S2-{arm}, which is not one of the
# pipeline's fixed names, hence --exp.
cd ..
EXP_FLAGS=""
for ARM in $ARMS; do
    EXP_FLAGS="$EXP_FLAGS --exp fcp-S2-${ARM}${EXP_SUFFIX}"
done
python extract_models/extract_S2_models.py --layout $LAYOUT --env Overcooked -a fcp $EXP_FLAGS || exit 1

# Stage-2 metrics are namespaced by the policy pair, hence the key prefix.
cd $PYTHONPATH/..
CURVE_ARMS=""
for ARM in $ARMS; do
    CURVE_ARMS="$CURVE_ARMS fcp-S2-${ARM}${EXP_SUFFIX}"
done
python experiments/fetch_training_curves.py --layout $LAYOUT \
    --arms $CURVE_ARMS --key_prefix "either-fcp_adaptive-" \
    --out experiments/results/training_curves_s2_${LAYOUT}.json

echo "stage 2 done. Now run: bash morl-benchmark-eval.slurm $LAYOUT \"1 2 3\" \"$ARMS\" \"$EXP_SUFFIX\""
