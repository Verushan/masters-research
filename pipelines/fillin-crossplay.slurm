#!/bin/bash -l
#SBATCH -J ZSCEvalFillInXP
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/fillin-crossplay-%j.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/fillin-crossplay-%j.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 24:00:00

# Fill-in suite, Step 8, last criterion: the standard zero-shot score.
#
# The Step 7 arms trained against scripted specialists, never against a learned
# partner. Whatever they gain at filling in, the neglect arm must not lose
# against the 16 held-out HSP partners -- the number every earlier arm was
# judged on. This cross-plays the three arms (10 seeds each) and, as a
# reference, the hand-shaped FCP stage-2 agent (fcp-S2-bench_sp) with the HSP
# pool, deterministic and stochastic, then runs the usual analysis.
#
# The env is built with the neglect arms' flags, so their weights move against
# each stranger as they did in training. Every other entry has a narrower
# observation and is trimmed back to it (EvalPolicy.fit_obs), and the weights do
# not reach any reward that is scored: the analysis reads ep_sparse_r.
#
# Usage: sbatch fillin-crossplay.slurm
#        sbatch --dependency=afterok:<training job> fillin-crossplay.slurm

cd /home-mscluster/vnaidoo/Coding/masters-research/pipelines

if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
fi
if [ "$SHOULD_SOURCE_CONDA" = "true" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi
conda activate zsceval

LAYOUT=${LAYOUT:-unident_s}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
ARMS="scripted-hand scripted-neglect scripted-noann"
REF_S2=${REF_S2:-bench_sp}
TASK_CONFIG=rnn_policy_config_mow-tasks.pkl
THREADS=12
DUMMY=2
EPISODES_STOCHASTIC=5
TAG=_hsp_scripted
RESULTS=$PYTHONPATH/../experiments/results

cd $PYTHONPATH/zsceval/scripts
python prep/store_policy_config.py $LAYOUT --morl_obs_weights tasks || exit 1

for A in $ARMS; do
    python extract_models/extract_S2_models.py --layout $LAYOUT --env Overcooked \
        --algorithm fcp --exp fcp-S2-${A} --checkpoint final \
        || echo "  stage-2 extraction failed for ${A}"
    echo "${A}: $(ls $POLICY_POOL/$LAYOUT/fcp/s2/fcp-S2-${A}/ 2>/dev/null | tr '\n' ' ')"
done

# The stage-1 slot is required by the generator; one bench_sp seed keeps the
# default stage-1 arms from being pulled in.
YML_OUT=$POLICY_POOL/$LAYOUT/morl_benchmark/cross_play${TAG}.yml
YML=$(python prep/gen_crossplay_yml.py $LAYOUT --arms bench_sp --arm_seeds 1 --peak_arms \
    --s2_arms $ARMS $REF_S2 --s2_arm_seeds $SEEDS \
    --s2_arm_config "scripted-neglect=${TASK_CONFIG}" "scripted-noann=${TASK_CONFIG}" \
    --heldout hsp --hsp_partners auto --out $YML_OUT --skip_missing | tail -1)
echo "population yml: $YML"
AGENTS=$(grep -oP '^[A-Za-z0-9_-]+(?=:)' $YML | tr '\n' ' ')
echo "agents ($(echo $AGENTS | wc -w)): $AGENTS"
echo "entries on the task-weights config: $(grep -c "$TASK_CONFIG" $YML)"
[ "$(grep -c "$TASK_CONFIG" $YML)" -gt 0 ] || { echo "no neglect-arm agents in the pool"; exit 1; }

mkdir -p $RESULTS
cd overcooked
common=(--env_name Overcooked --algorithm_name population --layout_name $LAYOUT
        --num_agents 2 --seed 1 --episode_length 400 --overcooked_version old
        --n_eval_rollout_threads $THREADS --dummy_batch_size $DUMMY
        --n_training_threads 2
        --use_morl --morl_objectives tasks --morl_weights 20,3,3,5 --morl_team_task
        --morl_adaptive_target neglect --use_morl_obs_weights
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
python experiments/analyze_crossplay.py \
    --records experiments/results/cross_play_${LAYOUT}${TAG}_deterministic.json \
              experiments/results/cross_play_${LAYOUT}${TAG}_stochastic.json \
    --partner_group heldout_hsp \
    --out experiments/results/metrics_${LAYOUT}${TAG}.json
echo "fill-in cross-play done for ${LAYOUT} at $(date)"
