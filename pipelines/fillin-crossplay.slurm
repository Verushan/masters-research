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
# The corrected-rule arms run as a separate matrix, because the neglect rule is
# an env setting and every agent in one matrix shares the env:
#   ARMS="scripted-hand scripted-neglect2 scripted-noann2" TAG=_hsp_scripted2 \
#   NEGLECT_EXTRA="--morl_neglect_prior 0.5 --morl_neglect_halflife 10" sbatch fillin-crossplay.slurm
#
# The no-swap ablation and the second kitchen reuse it:
#   ARMS="scripted-hand scripted-handns" TAG=_hsp_scripted_ns sbatch fillin-crossplay.slurm
#   LAYOUT=random3 ARMS="scripted-hand scripted-noann2" S1_SEEDS="1 2 3" TAG=_hsp_scripted \
#   NEGLECT_EXTRA="--morl_neglect_prior 0.5 --morl_neglect_halflife 10" sbatch fillin-crossplay.slurm
# SKIP_EXTRACT=1 scores what is already in the pool (fillin-train.slurm MODE=extract
# wrote it), so the fill-in suite can read the same files while this runs.
ARMS=${ARMS:-"scripted-hand scripted-neglect scripted-noann"}
# HELDOUT_ARGS / PARTNER_GROUP swap the held-out partner set. random1's HSP
# partners cooperate with nobody but their own training partner, so there the
# held-out set is fresh stage-1 self-play seeds no population was built from:
#   HELDOUT_ARGS="--heldout sp --heldout_exp bench_sp --heldout_seeds 7 8 9 10 11 12 --heldout_tags mid final" \
#   PARTNER_GROUP=heldout ...   (analyze_crossplay calls the self-play held-out group "heldout")
S1_SEEDS=${S1_SEEDS:-1}
NEGLECT_EXTRA=${NEGLECT_EXTRA:-}
REF_S2=${REF_S2:-bench_sp}
TASK_CONFIG=rnn_policy_config_mow-tasks.pkl
THREADS=12
DUMMY=2
EPISODES_STOCHASTIC=5
TAG=${TAG:-_hsp_scripted}
RESULTS=$PYTHONPATH/../experiments/results

cd $PYTHONPATH/zsceval/scripts
python prep/store_policy_config.py $LAYOUT --morl_obs_weights tasks || exit 1

for A in $ARMS; do
    [ -n "$SKIP_EXTRACT" ] || python extract_models/extract_S2_models.py --layout $LAYOUT --env Overcooked \
        --algorithm fcp --exp fcp-S2-${A} --checkpoint final \
        || echo "  stage-2 extraction failed for ${A}"
    echo "${A}: $(ls $POLICY_POOL/$LAYOUT/fcp/s2/fcp-S2-${A}/ 2>/dev/null | tr '\n' ' ')"
done

# The stage-1 slot is required by the generator; naming bench_sp keeps the
# default stage-1 arms from being pulled in. One seed on unident_s, where the
# reference is already scored; all of them on a kitchen scored for the first time.
CFG_FLAGS=()
for A in $ARMS; do
    # Only the neglect-weighted arms see the task weights; every other arm
    # (scripted-hand*, mixedspec-hand, bench_sp-annego, ...) loads plain.
    [[ "$A" == *neglect* || "$A" == *noann* ]] && CFG_FLAGS+=("${A}=${TASK_CONFIG}")
done
YML_OUT=$POLICY_POOL/$LAYOUT/morl_benchmark/cross_play${TAG}.yml
YML=$(python prep/gen_crossplay_yml.py $LAYOUT --arms bench_sp --arm_seeds $S1_SEEDS --peak_arms \
    --s2_arms $ARMS $REF_S2 --s2_arm_seeds $SEEDS \
    --s2_arm_config "${CFG_FLAGS[@]}" \
    ${HELDOUT_ARGS:---heldout hsp --hsp_partners auto} --out $YML_OUT --skip_missing | tail -1)
echo "population yml: $YML"
AGENTS=$(grep -oP '^[A-Za-z0-9_-]+(?=:)' $YML | tr '\n' ' ')
# HELDOUT_ONLY=1 plays every trained agent against the held-out partners only,
# in both seats, instead of the full agent x agent matrix. The zero-shot score
# needs nothing else, and for a large pool (the partner-removal experiment has
# 70+ trained agents) the full matrix costs three times as long. Self-play and
# agent-agent compatibility are then absent from the metrics.
EGOS="$AGENTS"
PARTNERS="$AGENTS"
ORDER_FLAG=()
if [ -n "$HELDOUT_ONLY" ]; then
    PARTNERS=$(echo $AGENTS | tr ' ' '\n' | grep '^heldout_' | tr '\n' ' ')
    EGOS=$(echo $AGENTS | tr ' ' '\n' | grep -v '^heldout_' | tr '\n' ' ')
    ORDER_FLAG=(--both_orders)
    echo "held-out only: $(echo $EGOS | wc -w) agents x $(echo $PARTNERS | wc -w) partners, both seats"
fi
echo "agents ($(echo $AGENTS | wc -w)): $AGENTS"
echo "entries on the task-weights config: $(grep -c "$TASK_CONFIG" $YML)"
if [ ${#CFG_FLAGS[@]} -gt 0 ] && [ "$(grep -c "$TASK_CONFIG" $YML)" -eq 0 ]; then
    echo "no neglect-arm agents in the pool"; exit 1
fi

mkdir -p $RESULTS
cd overcooked
common=(--env_name Overcooked --algorithm_name population --layout_name $LAYOUT
        --num_agents 2 --seed 1 --episode_length 400 --overcooked_version old
        --n_eval_rollout_threads $THREADS --dummy_batch_size $DUMMY
        --n_training_threads 2
        --use_morl --morl_objectives tasks --morl_weights 20,3,3,5 --morl_team_task
        --morl_adaptive_target neglect --use_morl_obs_weights $NEGLECT_EXTRA
        --population_yaml_path $YML
        --ego_policy_names $EGOS --partner_policy_names $PARTNERS "${ORDER_FLAG[@]}")
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
    --partner_group ${PARTNER_GROUP:-heldout_hsp} \
    --out experiments/results/metrics_${LAYOUT}${TAG}.json
echo "fill-in cross-play done for ${LAYOUT} at $(date)"
