#!/bin/bash -l
#SBATCH -J ZSCEvalBRPartners
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/br-partners-%A_%a.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/br-partners-%A_%a.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 72:00:00

# Train a best response to each held-out partner, to get a real BR-Prox
# denominator.
#
# analyze_crossplay.py currently uses the best score any agent already in the
# pool achieved against a partner. Nothing in the pool was trained to exploit
# anyone, so that is biased low and every number resting on it is optimistic in
# the same direction: BR-Prox overstates capture, regret understates the
# shortfall, CVaR understates the worst case.
#
# It also misled us about a layout. random3 had 13 of 16 partners at br_hat 0,
# which was read as "no headroom". It means only that nothing in our pool scored
# with them -- hsp12 paired with the w1 partner it trained beside returns 180.0,
# and w1 is never in the cross-play pool. The real reading is that every method
# captures approximately none of a large achievable return, which is a much
# stronger claim and makes random3 worth keeping rather than dropping.
#
# One run per partner, each against that single frozen partner
# (--population_size 1, --adaptive_agent_name br_agent). Budget is 2e6 to match
# the rest of this project rather than upstream's 1e7; a BR trained at a smaller
# budget is a *lower* bound on the best response, so BR-Prox computed from it is
# still optimistic -- less so than the pool-max proxy, and the report should say
# "proximity to the best response we could find" rather than imply otherwise.
#
# Usage: sbatch --array=0-1 br-partners.slurm      # one task per layout
#        LAYOUTS=random3 PARTNERS="12 13" sbatch --array=0 --export=ALL br-partners.slurm

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

LAYOUTS=${LAYOUTS:-"random0 random3"}
read -r -a LAYOUT_ARR <<< "$LAYOUTS"
LAYOUT=${LAYOUT_ARR[${SLURM_ARRAY_TASK_ID:-0}]}
STEPS=${STEPS:-2e6}
HSP_EXP=${HSP_EXP:-hsp-s1}
TAG=${TAG:-final}
EXP=${EXP:-br}
# .env exports a repo-wide ROLLOUT_THREADS and is loaded before this, so a
# ${ROLLOUT_THREADS:-N} default would never apply. That variable is the PPO
# batch size, so a BR trained at a different value than the agents it is the
# denominator for is not the same measurement.
BR_ROLLOUT_THREADS=${BR_ROLLOUT_THREADS:-12}
CONCURRENCY=${CONCURRENCY:-$(( ${SLURM_CPUS_PER_TASK:-14} / 7 ))}
[ "$CONCURRENCY" -lt 1 ] && CONCURRENCY=1

LOGDIR=/home-mscluster/vnaidoo/Coding/masters-research/experiments/logs/br/${LAYOUT}
mkdir -p "$LOGDIR"

case "$LAYOUT" in
    random0|random0_medium|random1|random3|small_corridor|unident_s) VERSION=old ;;
    *) VERSION=new ;;
esac

cd $PYTHONPATH/zsceval/scripts
PARTNERS=${PARTNERS:-$(python prep/gen_br_ymls.py $LAYOUT --hsp_exp $HSP_EXP --tag $TAG | tail -1)}
if [ -z "$PARTNERS" ]; then
    echo "no partners for ${LAYOUT}"
    exit 1
fi
echo "layout ${LAYOUT}, partners: ${PARTNERS}, steps ${STEPS}, concurrency ${CONCURRENCY}"

cd $PYTHONPATH/zsceval/scripts/overcooked

running=0
for i in $PARTNERS; do
    YML=$POLICY_POOL/${LAYOUT}/br/train_br_${i}.yml
    if [ ! -f "$YML" ]; then
        echo "### partner ${i}: no yml, skipping"
        continue
    fi
    echo "=== BR for partner ${i} ==="
    (
        python train/train_adaptive.py --env_name Overcooked --algorithm_name mappo \
            --experiment_name "${EXP}" --layout_name $LAYOUT --num_agents 2 \
            --seed $i --n_training_threads $TRAINING_THREADS \
            --n_rollout_threads $BR_ROLLOUT_THREADS --dummy_batch_size 1 \
            --episode_length 400 --num_env_steps $STEPS \
            --reward_shaping_horizon $STEPS \
            --overcooked_version $VERSION \
            --ppo_epoch 15 --entropy_coefs 0.2 0.05 0.01 \
            --entropy_coef_horizons 0 $(python -c "print(int(float('$STEPS')*0.25))") $STEPS \
            --stage 2 \
            --save_interval 25 --log_interval 10 \
            --use_eval --eval_interval 25 --n_eval_rollout_threads 12 \
            --eval_episodes 12 --eval_stochastic \
            --population_yaml_path "$YML" \
            --population_size 1 --adaptive_agent_name "br_agent" \
            --use_proper_time_limits \
            --wandb_tags br ${LAYOUT} \
            --wandb_name $WANDB_ENTITY
    ) > "${LOGDIR}/br_${i}.log" 2>&1 &

    running=$((running + 1))
    if [ "$running" -ge "$CONCURRENCY" ]; then
        wait -n 2>/dev/null || wait
        running=$((running - 1))
    fi
done
wait

echo "=========================================================="
echo "BR training done for ${LAYOUT} at $(date)"
echo "per-partner logs in ${LOGDIR}"
echo "=========================================================="
