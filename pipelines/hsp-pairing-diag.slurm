#!/bin/bash -l
#SBATCH -J ZSCEvalHSPPairDiag
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/hsp-pair-diag-%j.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/hsp-pair-diag-%j.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 02:00:00

# Why every random3 cross-play cell is zero, when the partners are not.
#
# The 16 random3 HSP bias agents each trained to 90-180 sparse return (W&B and
# the per-seed logs agree), which is comparable to random0's 55-200. Yet
# metrics_random3_hsp.json has every held-out partner at self_play 0.0 and 13 of
# 16 br_hat at 0. Partners that good should not produce a matrix that empty, so
# either the checkpoints in the pool are not the agents that trained, or the
# bias agents on this layout are so specialised that they only score with the
# exact w1 partner they were trained beside.
#
# This distinguishes the two. Each bias agent is a PAIR: w0 is the biased policy
# and w1 the plain sparse-reward partner trained alongside it, and w0+w1 is the
# only pairing either ever saw. So:
#
#   w0 + w1  ~= its training return  ->  checkpoints are fine; the zero matrix is
#                                        a real (if uninformative) coordination
#                                        result, and random3 is a floor effect
#   w0 + w1  ~= 0                    ->  the checkpoints in the pool are not the
#                                        trained agents; extraction is at fault
#
# random0 is run as the positive control: the same pairing there is known to
# work, so a zero on random0 would indict this script rather than random3.
#
# Usage: sbatch hsp-pairing-diag.slurm [layouts] [seeds]

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

LAYOUTS=${1:-"random3 random0"}
SEEDS=${2:-"12 40"}
EPISODES=${EPISODES:-5}

cd $PYTHONPATH/zsceval/scripts/overcooked
POOL=$PYTHONPATH/zsceval/scripts/overcooked/policy_pool
OUT=$PYTHONPATH/results/hsp_pairing_diag
mkdir -p "$OUT"

for LAYOUT in $LAYOUTS; do
    for SEED in $SEEDS; do
        W0="${LAYOUT}/hsp/s1/hsp-s1/hsp${SEED}_final_w0_actor.pt"
        W1="${LAYOUT}/hsp/s1/hsp-s1/hsp${SEED}_final_w1_actor.pt"
        if [ ! -f "$POOL/$W0" ] || [ ! -f "$POOL/$W1" ]; then
            echo "### ${LAYOUT} seed ${SEED}: checkpoints missing, skipping"
            continue
        fi

        YML="$POOL/${LAYOUT}/hsp_pair_diag_${SEED}.yml"
        # Bias agents are MLPs like every other stage-1 checkpoint; the shared
        # runner's recurrent flag is inverted, see CLAUDE.md.
        cat > "$YML" <<YAMLEOF
hsp${SEED}_w0:
    policy_config_path: ${LAYOUT}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: ${W0}
hsp${SEED}_w1:
    policy_config_path: ${LAYOUT}/policy_config/mlp_policy_config.pkl
    featurize_type: ppo
    train: False
    model_path:
        actor: ${W1}
YAMLEOF

        echo "########## ${LAYOUT} seed ${SEED}: w0 + w1 ##########"
        python eval/eval.py \
            --env_name Overcooked \
            --algorithm_name population \
            --experiment_name hsp_pair_diag \
            --layout_name "$LAYOUT" \
            --num_agents 2 \
            --seed 1 \
            --episode_length 400 \
            --n_eval_rollout_threads 1 \
            --eval_episodes "$EPISODES" \
            --population_yaml_path "$YML" \
            --population_size 2 \
            --dummy_batch_size 1 \
            --agent0_policy_name "hsp${SEED}_w0" \
            --agent1_policy_name "hsp${SEED}_w1" \
            --overcooked_version old \
            --eval_result_path "${OUT}/${LAYOUT}_hsp${SEED}_w0w1.json" \
            || echo "  FAILED ${LAYOUT} seed ${SEED}"

        echo "########## ${LAYOUT} seed ${SEED}: w0 + w0 (the metric's self_play) ##########"
        python eval/eval.py \
            --env_name Overcooked \
            --algorithm_name population \
            --experiment_name hsp_pair_diag \
            --layout_name "$LAYOUT" \
            --num_agents 2 \
            --seed 1 \
            --episode_length 400 \
            --n_eval_rollout_threads 1 \
            --eval_episodes "$EPISODES" \
            --population_yaml_path "$YML" \
            --population_size 2 \
            --dummy_batch_size 1 \
            --agent0_policy_name "hsp${SEED}_w0" \
            --agent1_policy_name "hsp${SEED}_w0" \
            --overcooked_version old \
            --eval_result_path "${OUT}/${LAYOUT}_hsp${SEED}_w0w0.json" \
            || echo "  FAILED ${LAYOUT} seed ${SEED} (w0+w0)"
    done
done

echo "=========================================================="
echo "results in ${OUT}"
grep -H . "${OUT}"/*.json 2>/dev/null | head -40
echo "=========================================================="
