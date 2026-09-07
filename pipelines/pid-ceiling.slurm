#!/bin/bash -l
#SBATCH -J ZSCEvalPIDCeiling
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/pid-ceiling-%j.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/pid-ceiling-%j.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 8:00:00

# The oracle ceiling: does knowing who your partner is help at all?
#
# This is the decisive experiment for the adaptation half of the project, and
# it is cheap. Each ladder rung is scored against *its own training
# population*, with the true partner ids -- not against held-out partners, who
# have no id the agent ever saw. Scoring a rung on held-out partners measures
# the handicap of a useless input, not the value of a useful one, which is why
# the existing pid-ladder-eval.slurm could not answer the question even once
# the rungs were extractable.
#
#   rung1  actor never sees the id (control)
#   rung2  actor sees the raw scalar id
#   rung3  actor sees a one-hot over the population
#
# If rung3 does not beat rung1 here, then no amount of partner inference can
# help on this layout -- the information is available and worthless -- and the
# project's adaptation claim should become a robustness claim. If it does beat
# rung1, the gap is the headroom any adaptation method is competing for, and
# the question becomes whether identity can be inferred from behaviour.
#
# Every rung plays the same partners under the same budget, so the only
# difference between them is what the actor is shown.
#
# Usage: sbatch pid-ceiling.slurm [layout] [arm] [seeds] [rungs]

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

LAYOUT=${1:-random0}
ARM=${2:-bench_sp}
# Seed *labels*, not seed numbers: rung1 and rung3 each have two finished
# runs of seed 1, which --replicates keeps as `1` and `1r2`. A label with no
# checkpoint is skipped, so the same list covers all three rungs.
SEEDS=${3:-"1 1r2 2 3"}
RUNGS=${4:-"rung1 rung2 rung3"}
THREADS=9
DUMMY=3
EPISODES=5
RESULTS=$PYTHONPATH/../experiments/results
TRAIN_YML_REL=${LAYOUT}/fcp/s2/train-${ARM}.yml
TRAIN_YML=$POLICY_POOL/${TRAIN_YML_REL}

if [ ! -f "$TRAIN_YML" ]; then
    echo "missing ${TRAIN_YML} -- run prep/gen_arm_S2_yml.py ${LAYOUT} --arm ${ARM} first"
    exit 1
fi

# The one-hot width is the training yml's entry count, which is what
# train_morl_stage_2.sh used when it trained rung3.
POP=$(grep -cE "^[a-zA-Z_][a-zA-Z_0-9]*:" "$TRAIN_YML")
echo "population yml ${TRAIN_YML_REL}: ${POP} entries"

cd $PYTHONPATH/zsceval/scripts

# Configs at the rungs' own observation widths. env_policy trims each frozen
# partner back to the 20 channels it was built for.
python prep/store_policy_config.py $LAYOUT --pid_obs_dim 0   || exit 1
python prep/store_policy_config.py $LAYOUT --pid_obs_dim $POP || exit 1

for RUNG in $RUNGS; do
    echo "########## extract ${ARM}-${RUNG} ##########"
    python extract_models/extract_S2_models.py --layout $LAYOUT --env Overcooked \
        --algorithm fcp --exp fcp-S2-${ARM}-${RUNG} --checkpoint final --replicates \
        || echo "  extraction failed for ${RUNG}"
done

PARTNERS=$(grep -oE "^[a-zA-Z_][a-zA-Z_0-9]*:" "$TRAIN_YML" | tr -d ':' | grep -v '^fcp_adaptive$' | tr '\n' ' ')
echo "partners: ${PARTNERS}"
mkdir -p $RESULTS

for RUNG in $RUNGS; do
    case "$RUNG" in
        rung1) PID_FLAGS=(); PID_DIM_ARG=() ;;
        rung2) PID_FLAGS=(--use_agent_policy_id_obs --agent_policy_id_obs_dim 0)
               PID_DIM_ARG=(--pid_obs_dim 0) ;;
        rung3) PID_FLAGS=(--use_agent_policy_id_obs --agent_policy_id_obs_dim $POP)
               PID_DIM_ARG=(--pid_obs_dim $POP) ;;
        *) echo "unknown rung ${RUNG}"; exit 1 ;;
    esac

    for SEED in $SEEDS; do
        ACTOR_REL=${LAYOUT}/fcp/s2/fcp-S2-${ARM}-${RUNG}/${SEED}.pt
        if [ ! -f "$POLICY_POOL/$ACTOR_REL" ]; then
            echo "### ${RUNG} seed ${SEED}: no checkpoint, skipping"
            continue
        fi
        YML_REL=${LAYOUT}/pid_ladder/${ARM}_${RUNG}_s${SEED}.yml
        python prep/gen_pid_eval_yml.py $LAYOUT --train_yml "$TRAIN_YML_REL" \
            --actor "$ACTOR_REL" "${PID_DIM_ARG[@]}" --out "$YML_REL" || exit 1

        echo "########## ${RUNG} seed ${SEED} vs training partners ##########"
        (cd overcooked && python eval/cross_play.py \
            --env_name Overcooked --algorithm_name population --layout_name $LAYOUT \
            --num_agents 2 --seed 1 --episode_length 400 --overcooked_version old \
            --n_eval_rollout_threads $THREADS --dummy_batch_size $DUMMY \
            --n_training_threads 2 --morl_objectives default \
            --population_yaml_path "$POLICY_POOL/$YML_REL" \
            --ego_policy_names fcp_adaptive --partner_policy_names $PARTNERS \
            "${PID_FLAGS[@]}" \
            --experiment_name pidceil_${LAYOUT}_${RUNG}_s${SEED} \
            --eval_stochastic --eval_episodes $EPISODES \
            --eval_result_path $RESULTS/pid_ceiling_${LAYOUT}_${ARM}_${RUNG}_s${SEED}.json) \
            || echo "  FAILED ${RUNG} seed ${SEED}"
    done
done

echo "=========================================================="
echo "PID CEILING done for ${LAYOUT}/${ARM} at $(date)"
ls -la $RESULTS/pid_ceiling_${LAYOUT}_${ARM}_*.json 2>/dev/null
echo "=========================================================="
