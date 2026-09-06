#!/bin/bash -l
#SBATCH -J ZSCEvalMORLBenchEval
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/morl-benchmark-eval-%j.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/morl-benchmark-eval-%j.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 8:00:00

# MORL benchmark, evaluation half. Run after morl-benchmark.slurm has trained
# and extracted the four arms.
#
# Builds one evaluation pool holding the benchmark arms, the FCP stage-2 agent
# and the held-out stage-1 partners, then walks the full cross-play matrix twice:
#
#   deterministic  the canonical ZSC-Eval number. The start state is fixed and
#                  actions are argmax, so a pair replays identically -- one
#                  episode per cell is the whole distribution, and repeating it
#                  would only buy duplicate rows.
#   stochastic     actions sampled from the policy. This is the only pass that
#                  can report the proposal's "return stability", since it is the
#                  only one where repeated rollouts of one pair differ.
#
# `--morl_objectives default` makes every cell report the objective breakdown as
# well as the return, which is what keeps the comparison legible on layouts where
# cross-play sparse return collapses to zero for every method.
#
# Usage: bash morl-benchmark-eval.slurm [layout] [arm_seeds] [s2_arms] [s2_suffix]
#
# The third argument adds the stage-2 agents trained by morl-s2.slurm to the
# pool (`fcp-S2-{arm}`), so the stage-1 arms and the FCP agents built from them
# are scored in one matrix against the same held-out partners. Results then go
# to *_s2_* files rather than overwriting the stage-1-only run the current
# report is built from.
#
# HELDOUT picks the partner set the ZSC number is averaged over. `sp` is the
# pre-existing self-play pool and exists for random0 only, which is why it was
# the default while random0 was the only layout: on any other layout every `sp`
# entry is silently dropped by --skip_missing and the matrix comes out with no
# held-out column at all. `hsp` is the bias-agent set hsp-partners.slurm trains,
# which exists for every layout. The two are never merged -- averaging a ZSC
# metric over bias agents and self-play checkpoints together would average two
# different evaluations.
#
#   HELDOUT=hsp bash morl-benchmark-eval.slurm unident_s "1 2 3"

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

LAYOUT=${1:-random0}
ARM_SEEDS=${2:-"1 2 3"}
S2_ARMS=${3:-""}
HELDOUT=${HELDOUT:-sp}
HSP_EXP=${HSP_EXP:-hsp-s1}
# Stage-2 seed labels. Left empty, gen_crossplay_yml uses its own default of
# 1 2 3. Set it when `extract_S2_models.py --replicates` has recovered repeat
# runs of a seed, which land as `1r2.pt`, `2r2.pt` and so on: pass the union of
# every arm's labels and let --skip_missing drop the ones an arm does not have.
#
#   S2_ARM_SEEDS="1 2 3 1r2 2r2 3r2" HELDOUT=hsp bash morl-benchmark-eval.slurm ...
S2_ARM_SEEDS=${S2_ARM_SEEDS:-""}
# Experiment-name suffix the stage-2 runs used, e.g. "-pilot". Must match.
S2_SUFFIX=${4:-""}
THREADS=12
# Envs per worker process, not a minibatch size: dummy_batch_size == THREADS
# would put every env in one process and serialise the whole matrix.
DUMMY=2
EPISODES_STOCHASTIC=5
RESULTS=$PYTHONPATH/../experiments/results

cd $PYTHONPATH/zsceval/scripts

if [ -n "$S2_ARMS" ]; then
    S2_FLAG="--s2_arms $S2_ARMS"
    # Appended only when set: an empty --s2_suffix would swallow the next flag.
    # The `=` form is required, not cosmetic -- suffixes conventionally start with
    # a dash ("-pilot"), and argparse reads a space-separated "-pilot" as a flag
    # rather than as this option's value.
    [ -n "$S2_SUFFIX" ] && S2_FLAG="$S2_FLAG --s2_suffix=$S2_SUFFIX"
    [ -n "$S2_ARM_SEEDS" ] && S2_FLAG="$S2_FLAG --s2_arm_seeds $S2_ARM_SEEDS"
    TAG="_s2"
else
    S2_FLAG=""
    TAG=""
fi

# Only the non-default partner set changes the file names, so a `sp` run still
# writes exactly the paths the existing random0 results live at.
if [ "$HELDOUT" = "sp" ]; then
    HELDOUT_FLAG="--heldout sp"
else
    HELDOUT_FLAG="--heldout $HELDOUT --hsp_partners auto --hsp_exp $HSP_EXP"
    TAG="${TAG}_${HELDOUT}"
fi

# One yml per (layout, pool shape): the default path is shared, so an hsp run
# would otherwise overwrite the sp pool the earlier results were produced from.
YML_OUT=$POLICY_POOL/$LAYOUT/morl_benchmark/cross_play${TAG}.yml
YML=$(python prep/gen_crossplay_yml.py $LAYOUT --arm_seeds $ARM_SEEDS $S2_FLAG \
    $HELDOUT_FLAG --out $YML_OUT --skip_missing | tail -1)
echo "population yml: $YML"

# Every policy in the pool plays every other, in both seats.
AGENTS=$(grep -oP '^[A-Za-z0-9_]+(?=:)' $YML | tr '\n' ' ')
echo "agents: $AGENTS"

mkdir -p $RESULTS
cd overcooked

common=(--env_name Overcooked --algorithm_name population --layout_name $LAYOUT
        --num_agents 2 --seed 1 --episode_length 400 --overcooked_version old
        --n_eval_rollout_threads $THREADS --dummy_batch_size $DUMMY
        --n_training_threads 2 --morl_objectives default
        --population_yaml_path $YML
        --ego_policy_names $AGENTS --partner_policy_names $AGENTS)

python eval/cross_play.py "${common[@]}" \
    --experiment_name xp_${LAYOUT}${TAG}_det \
    --eval_episodes 1 \
    --eval_result_path $RESULTS/cross_play_${LAYOUT}${TAG}_deterministic.json || exit 1

python eval/cross_play.py "${common[@]}" \
    --experiment_name xp_${LAYOUT}${TAG}_sto \
    --eval_stochastic --eval_episodes $EPISODES_STOCHASTIC \
    --eval_result_path $RESULTS/cross_play_${LAYOUT}${TAG}_stochastic.json || exit 1

cd $PYTHONPATH/..
python experiments/fetch_training_curves.py --layout $LAYOUT \
    --out experiments/results/training_curves_${LAYOUT}.json
# group_of() names the sp pool's partners "heldout" and the bias agents
# "heldout_hsp", and analyze_crossplay.py asserts rather than reporting an
# average over nothing -- but only after both cross-play passes have been paid
# for, so the group has to match the pool the yml was built from.
if [ "$HELDOUT" = "sp" ]; then
    PARTNER_GROUP=heldout
else
    PARTNER_GROUP=heldout_$HELDOUT
fi
python experiments/analyze_crossplay.py \
    --records experiments/results/cross_play_${LAYOUT}${TAG}_deterministic.json \
              experiments/results/cross_play_${LAYOUT}${TAG}_stochastic.json \
    --partner_group $PARTNER_GROUP \
    --out experiments/results/metrics_${LAYOUT}${TAG}.json

# The per-episode dumps are the expensive artifact -- ~45 min of rollouts -- and
# compress about 35x, so they are kept gzipped. analyze_crossplay.py reads either.
gzip -9 -f experiments/results/cross_play_${LAYOUT}${TAG}_deterministic.json \
           experiments/results/cross_play_${LAYOUT}${TAG}_stochastic.json
