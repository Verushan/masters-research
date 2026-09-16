#!/bin/bash -l
#SBATCH -J ZSCEvalObsWEval
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/obs-weights-eval-%j.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/obs-weights-eval-%j.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 8:00:00

# Cross-play the observable-preference arm (bench_morl_ad_obs-obs) against the
# held-out HSP partners, alongside the arms it is meant to improve on.
#
# This arm cannot go through morl-benchmark-eval.slurm as it stands. It was
# trained with --use_morl_obs_weights, which appends the live preference vector
# w to the observation as K channels, so:
#
#   * its checkpoint only loads through a policy config written at that width
#     (prep/store_policy_config.py --morl_obs_weights SET), not the shared
#     mlp_policy_config.pkl every other pool member is rebuilt from;
#   * the cross-play env has to be built with the same MORL flags the arm was
#     trained under -- --use_morl, the objective set, --morl_adaptive_weights
#     and --use_morl_obs_weights -- so that w exists, moves the way it did in
#     training, and is appended to the observation. Every frozen partner is
#     trimmed back to the width it was trained on (EvalPolicy.fit_obs), exactly
#     as the partner-conditioning ladder is cross-played.
#
# The MORL reward the env computes is never used here -- the ZSC number is
# eval_ep_sparse_r from the base env -- but the adaptive w is: this is the one
# evaluation in which the adapting agent can see the preference it is adapting,
# and in which w moves in response to a *stranger's* behaviour.
#
# Usage:
#   sbatch obs-weights-eval.slurm random0           # anchored set
#   sbatch obs-weights-eval.slurm unident_s         # anchored_live3 set
#   OBS_ARM=bench_morl_ad_obs-obs COMPARE_ARMS="bench_sp bench_morl_ad-anc" sbatch ...
#
# The partner-conditioned arms use the same mechanism with a different set of
# widening flags. The channels are appended in a fixed order (w, then shares),
# so a shares-only policy cannot be trimmed out of a w+shares observation: each
# widened arm gets its own run. VARIANT picks the config suffix, the flags the
# env is built with, and the output tag:
#
#   VARIANT=fill   sbatch obs-weights-eval.slurm unident_s   # w + shares, complement rule
#   VARIANT=shares sbatch obs-weights-eval.slurm unident_s   # shares only, baseline reward
#
# S2_ARMS adds stage-2 trainees to the pool. A MORL ego trained with the fill
# flags has the widened observation and needs the rnn config at that width;
# S2_WIDE_ARMS names which of them do. OUT_TAG keeps the outputs apart.
#
#   VARIANT=fill S2_ARMS="bench_sp-annego bench_sp-fillego" S2_WIDE_ARMS="bench_sp-fillego" \
#     OUT_TAG=_s2ego sbatch obs-weights-eval.slurm unident_s

cd /home-mscluster/vnaidoo/Coding/masters-research/pipelines

if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
fi
if [ "$SHOULD_SOURCE_CONDA" = "true" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi
conda activate zsceval

LAYOUT=${1:-random0}
SEEDS=${2:-"1 2 3 4 5 6"}
OBS_ARM=${OBS_ARM:-bench_morl_ad_obs-obs}
# The objective set the arm was trained on, and so the width of its w. It
# follows the layout the same way rebaseline-anchored.sh picked it: unident_s
# has no counter handoffs, so its arms use the three-objective set.
case "$LAYOUT" in
    unident_s) OBJ=${OBJ:-anchored_live3}; UNIFORM_W="0.3333,0.3333,0.3333"
               COMPARE_ARMS=${COMPARE_ARMS:-"bench_sp bench_sparse bench_morl-live3 bench_morl_ad-live3"} ;;
    *)         OBJ=${OBJ:-anchored};       UNIFORM_W="0.25,0.25,0.25,0.25"
               COMPARE_ARMS=${COMPARE_ARMS:-"bench_sp bench_sparse bench_morl-anc bench_morl_ad-anc"} ;;
esac
case "$LAYOUT" in
    random0|random0_medium|random1|random3|small_corridor|unident_s) VERSION=old; WB_PROJECT=Overcooked ;;
    *) VERSION=new; WB_PROJECT=Overcooked-new ;;
esac
THREADS=12
DUMMY=2
EPISODES_STOCHASTIC=5
RESULTS=$PYTHONPATH/../experiments/results
VARIANT=${VARIANT:-weights}
case "$VARIANT" in
    weights)
        TAG="_hsp_obs"
        STORE_FLAGS=(--morl_obs_weights $OBJ)
        MOW_CONFIG="mlp_policy_config_mow-${OBJ}.pkl"
        ENV_FLAGS=(--use_morl --morl_objectives $OBJ --morl_weights "$UNIFORM_W"
                   --morl_adaptive_weights --use_morl_obs_weights) ;;
    fill)
        TAG="_hsp_fill"
        OBS_ARM=${OBS_ARM_OVERRIDE:-bench_morl_fill-live3}
        STORE_FLAGS=(--morl_obs_weights $OBJ --morl_obs_shares $OBJ)
        MOW_CONFIG="mlp_policy_config_mow-${OBJ}_mos.pkl"
        # Exactly the flags train_morl_benchmark.sh gives bench_morl_fill, so the
        # per-agent w moves against each stranger the way it did in training.
        ENV_FLAGS=(--use_morl --morl_objectives $OBJ --morl_weights "${ANN_WEIGHTS:-20,3,3}"
                   --morl_anneal_dense --morl_adaptive_weights --morl_adaptive_target complement
                   --use_morl_obs_weights --use_morl_obs_shares) ;;
    shares)
        TAG="_hsp_shares"
        OBS_ARM=${OBS_ARM_OVERRIDE:-bench_sp_shares-live3}
        STORE_FLAGS=(--morl_obs_shares $OBJ)
        MOW_CONFIG="mlp_policy_config_mos-${OBJ}.pkl"
        ENV_FLAGS=(--morl_objectives $OBJ --use_morl_obs_shares) ;;
    *) echo "unknown VARIANT ${VARIANT}"; exit 1 ;;
esac
TAG="${TAG}${OUT_TAG:-}"
echo "layout ${LAYOUT}  objectives ${OBJ}  obs arm ${OBS_ARM}  vs ${COMPARE_ARMS}  s2 ${S2_ARMS:-none}"

cd $PYTHONPATH/zsceval/scripts

# 1. A policy config at the widened observation, e.g. random0 (5,5,24).
python prep/store_policy_config.py $LAYOUT "${STORE_FLAGS[@]}" || exit 1
[ -f "$POLICY_POOL/$LAYOUT/policy_config/$MOW_CONFIG" ] || { echo "missing $MOW_CONFIG"; exit 1; }

# 2. The arm's checkpoints, ranked by its own reward -- ep_morl_r for a MORL
#    arm, ep_sparse_r for the baseline-reward shares arm, which logs no ep_morl_r.
case "$OBS_ARM" in bench_sp*) RANK=ep_sparse_r ;; *) RANK=ep_morl_r ;; esac
echo "########## extract ${OBS_ARM} (${RANK}) ##########"
python extract_models/extract_sp_models.py "$LAYOUT" "$WB_PROJECT" "$OBS_ARM" $RANK \
    || echo "  extraction failed for ${OBS_ARM}"
# Standard-width arms trained alongside (the annealed uniform-w arm) join the
# pool of every variant; extracted here if not yet in the pool.
for EXTRA in ${EXTRA_ARMS:-}; do
    if [ "$(ls $POLICY_POOL/$LAYOUT/fcp/s1/$EXTRA/*_final_actor.pt 2>/dev/null | wc -l)" -eq 0 ]; then
        python extract_models/extract_sp_models.py "$LAYOUT" "$WB_PROJECT" "$EXTRA" ep_morl_r \
            || echo "  extraction failed for ${EXTRA}"
    fi
    COMPARE_ARMS="$COMPARE_ARMS $EXTRA"
done
N=$(ls $POLICY_POOL/$LAYOUT/fcp/s1/$OBS_ARM/*_final_actor.pt 2>/dev/null | wc -l)
echo "${OBS_ARM}: ${N} final checkpoints"
[ "$N" -gt 0 ] || exit 1

# 2b. Stage-2 trainees, if any: extract them and point the widened ones at the
#     rnn config store_policy_config wrote alongside the mlp one.
S2_FLAGS=()
if [ -n "${S2_ARMS:-}" ]; then
    for A in $S2_ARMS; do
        python extract_models/extract_S2_models.py --layout $LAYOUT --env Overcooked \
            --algorithm fcp --exp fcp-S2-${A} --checkpoint final \
            || echo "  stage-2 extraction failed for ${A}"
    done
    S2_FLAGS=(--s2_arms $S2_ARMS --s2_arm_seeds $SEEDS)
    RNN_CONFIG=${MOW_CONFIG/mlp_/rnn_}
    for A in ${S2_WIDE_ARMS:-}; do
        S2_FLAGS+=(--s2_arm_config "${A}=${RNN_CONFIG}")
    done
fi

# 3. The pool. --skip_missing drops a comparison arm that does not exist on this
#    layout rather than failing; the obs arm itself was checked above.
YML_OUT=$POLICY_POOL/$LAYOUT/morl_benchmark/cross_play${TAG}.yml
YML=$(python prep/gen_crossplay_yml.py $LAYOUT --arm_seeds $SEEDS \
    --arms $COMPARE_ARMS $OBS_ARM --peak_arms \
    --arm_config "${OBS_ARM}=${MOW_CONFIG}" "${S2_FLAGS[@]}" \
    --heldout hsp --hsp_partners auto --out $YML_OUT --skip_missing | tail -1)
echo "population yml: $YML"
AGENTS=$(grep -oP '^[A-Za-z0-9_-]+(?=:)' $YML | tr '\n' ' ')
echo "agents: $AGENTS"
grep -c "$MOW_CONFIG" $YML | xargs echo "entries on the widened config:"

mkdir -p $RESULTS
cd overcooked

# 4. Cross-play with the env built the way the obs arm was trained.
common=(--env_name Overcooked --algorithm_name population --layout_name $LAYOUT
        --num_agents 2 --seed 1 --episode_length 400 --overcooked_version $VERSION
        --n_eval_rollout_threads $THREADS --dummy_batch_size $DUMMY
        --n_training_threads 2
        "${ENV_FLAGS[@]}"
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
echo "=========================================================="
echo "obs-weights eval done for ${LAYOUT} at $(date)"
echo "=========================================================="
