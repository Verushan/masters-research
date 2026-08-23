#!/bin/bash -l
#SBATCH -J ZSCEvalMORLBenchEval
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/morl-benchmark-eval.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/morl-benchmark-eval.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 16
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
THREADS=10
EPISODES_STOCHASTIC=8
RESULTS=$PYTHONPATH/../experiments/results

cd $PYTHONPATH/zsceval/scripts

YML=$(python prep/gen_crossplay_yml.py $LAYOUT --arm_seeds $ARM_SEEDS --skip_missing | tail -1)
echo "population yml: $YML"

# Every policy in the pool plays every other, in both seats.
AGENTS=$(grep -oP '^[A-Za-z0-9_]+(?=:)' $YML | tr '\n' ' ')
echo "agents: $AGENTS"

mkdir -p $RESULTS
cd overcooked

common=(--env_name Overcooked --algorithm_name population --layout_name $LAYOUT
        --num_agents 2 --seed 1 --episode_length 400 --overcooked_version old
        --n_eval_rollout_threads $THREADS --dummy_batch_size $THREADS
        --n_training_threads 2 --morl_objectives default
        --population_yaml_path $YML
        --ego_policy_names $AGENTS --partner_policy_names $AGENTS)

python eval/cross_play.py "${common[@]}" \
    --experiment_name xp_${LAYOUT}_det \
    --eval_episodes 1 \
    --eval_result_path $RESULTS/cross_play_${LAYOUT}_deterministic.json || exit 1

python eval/cross_play.py "${common[@]}" \
    --experiment_name xp_${LAYOUT}_sto \
    --eval_stochastic --eval_episodes $EPISODES_STOCHASTIC \
    --eval_result_path $RESULTS/cross_play_${LAYOUT}_stochastic.json || exit 1

cd $PYTHONPATH/..
python experiments/fetch_training_curves.py --layout $LAYOUT \
    --out experiments/results/training_curves_${LAYOUT}.json
python experiments/analyze_crossplay.py \
    --records experiments/results/cross_play_${LAYOUT}_deterministic.json \
              experiments/results/cross_play_${LAYOUT}_stochastic.json \
    --out experiments/results/metrics_${LAYOUT}.json
