#!/bin/bash -l

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

LAYOUT="random0"

cd $PYTHONPATH/zsceval/scripts/overcooked

yaml_path="$PYTHONPATH/zsceval/scripts/overcooked/policy_pool/random0/fcp/s2/train-s16-sp-1.yml"

python eval/eval.py \
    --env_name Overcooked \
    --algorithm_name population \
    --experiment_name custom_render \
    --layout_name $LAYOUT \
    --num_agents 2 \
    --seed 1 --use_wandb \
    --episode_length 400 \
    --n_eval_rollout_threads 1 \
    --eval_episodes 1 \
    --use_render \
    --population_yaml_path ${yaml_path} \
    --population_size 2 \
    --dummy_batch_size 1 \
    --agent0_policy_name sp14_3 \
    --agent1_policy_name sp16_3 \
    --overcooked_version old \
    --eval_result_path ${PYTHONPATH}/results/$LAYOUT/custom_render.json