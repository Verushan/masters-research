#!/bin/bash -l
#SBATCH -J ZSCEvalMORLBench
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/morl-benchmark-%j.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/morl-benchmark-%j.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 36:00:00

# MORL benchmark, stage 1.
#
# Trains four self-play arms that differ *only* in the reward the PPO buffer
# sees, then extracts init/mid/final checkpoints for each into the policy pool
# so they can be cross-played against the existing ZSC-Eval agents.
#
#   bench_sp        sparse + hand-shaped   (ZSC-Eval baseline)
#   bench_sparse    sparse only            (control: no dense signal)
#   bench_morl      w . r_vec, fixed w     (proposed)
#   bench_morl_ad   w . r_vec, adaptive w  (proposal 4.2.3)
#
# random0 is "forced coordination": the two agents are separated by a counter
# column, so every onion and dish must be handed across it. That makes it the
# layout where the `coordination` objective -- which the hand-crafted shaping
# does not reward at all -- has the most to say.
#
# The layout is a positional argument, so one submission per layout. Logs are
# keyed by %j because two layouts submitted concurrently would otherwise write
# the same file -- there is no array index here to tell them apart.

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

# Usage: sbatch morl-benchmark.slurm [layout] [seed_begin] [seed_max] [arms]
#
# The fourth argument narrows the run to a subset of the arms, so a seed range
# can be split across concurrent jobs instead of training 4 x N runs in series
# inside one wall clock. Extraction is restricted to the same subset, and since
# it keys checkpoints by seed it adds to the pool rather than replacing it.
LAYOUT=${1:-random0}
SEED_BEGIN=${2:-1}
SEED_MAX=${3:-3}
ARMS=${4:-"bench_sp bench_sparse bench_morl bench_morl_ad"}

# Same trap HSP_ROLLOUT_THREADS exists for: .env exports a repo-wide
# ROLLOUT_THREADS (16 on the cluster, 4 on the workstation) and is loaded above,
# so train_morl_benchmark.sh would take whichever value the machine happens to
# carry. That is the PPO batch size. The random0 arms were trained at 12 on the
# workstation, so a cluster run at 16 produces arms that are NOT comparable with
# them -- and a cross-layout comparison is the whole point of running this on a
# second layout. Pin it, and override deliberately if it ever needs to change.
export ROLLOUT_THREADS=${MORL_ROLLOUT_THREADS:-12}

cd $PYTHONPATH/zsceval/scripts/overcooked

# Cheap offline assertion that the MORL reward path is what we think it is.
python morl/check_morl_reward.py || exit 1

for ARM in $ARMS; do
    bash shell/train_morl_benchmark.sh $LAYOUT $ARM $SEED_BEGIN $SEED_MAX || exit 1
done

# The MORL arms are ranked by the reward they actually optimised; the other two
# by sparse return.
cd ..
for ARM in $ARMS; do
    case "$ARM" in
        bench_morl|bench_morl_ad) METRIC=ep_morl_r ;;
        *)                        METRIC=ep_sparse_r ;;
    esac
    python extract_models/extract_sp_models.py $LAYOUT overcooked $ARM $METRIC
done

# The MORL arms' own reward and the task's return can come apart -- a run whose
# scalarized reward keeps climbing while deliveries stop would otherwise be
# benchmarked only on the policy it collapsed to. Also take the checkpoint where
# sparse return actually peaked, so the report can separate "what this reward
# reached" from "where it ended up".
PEAK_EXPS=""
for ARM in $ARMS; do
    case "$ARM" in bench_morl|bench_morl_ad) PEAK_EXPS="$PEAK_EXPS --exp $ARM" ;; esac
done
[ -n "$PEAK_EXPS" ] && python extract_models/extract_peak_models.py \
    --layout $LAYOUT --env Overcooked $PEAK_EXPS --metric ep_sparse_r
