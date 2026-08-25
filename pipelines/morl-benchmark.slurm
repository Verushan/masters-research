#!/bin/bash -l
#SBATCH -J ZSCEvalMORLBench
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/morl-benchmark.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/morl-benchmark.err
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

# Usage: sbatch morl-benchmark.slurm [layout] [seed_begin] [seed_max]
LAYOUT=${1:-random0}
SEED_BEGIN=${2:-1}
SEED_MAX=${3:-3}

cd $PYTHONPATH/zsceval/scripts/overcooked

# Cheap offline assertion that the MORL reward path is what we think it is.
python morl/check_morl_reward.py || exit 1

for ARM in bench_sp bench_sparse bench_morl bench_morl_ad; do
    bash shell/train_morl_benchmark.sh $LAYOUT $ARM $SEED_BEGIN $SEED_MAX || exit 1
done

# The MORL arms are ranked by the reward they actually optimised; the other two
# by sparse return.
cd ..
python extract_models/extract_sp_models.py $LAYOUT overcooked bench_sp      ep_sparse_r
python extract_models/extract_sp_models.py $LAYOUT overcooked bench_sparse  ep_sparse_r
python extract_models/extract_sp_models.py $LAYOUT overcooked bench_morl    ep_morl_r
python extract_models/extract_sp_models.py $LAYOUT overcooked bench_morl_ad ep_morl_r

# The MORL arms' own reward and the task's return can come apart -- a run whose
# scalarized reward keeps climbing while deliveries stop would otherwise be
# benchmarked only on the policy it collapsed to. Also take the checkpoint where
# sparse return actually peaked, so the report can separate "what this reward
# reached" from "where it ended up".
python extract_models/extract_peak_models.py --layout $LAYOUT --env Overcooked \
    --exp bench_morl --exp bench_morl_ad --metric ep_sparse_r
