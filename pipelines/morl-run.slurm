#!/bin/bash -l
#SBATCH -J ZSCEvalMORL
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/morl-run.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/morl-run.err
#SBATCH --partition=bigbatch
#SBATCH -n 1                   # Number of tasks (processes)
#SBATCH -c 16                  # Number of CPU cores per task
#SBATCH -t 36:00:00            # Max runtime (hh:mm:ss)

# Stage 1 for the MORL agent: self-play trained on the scalarized objective
# vector instead of sparse + shaped reward. Same shape as sp-run.slurm, but the
# checkpoints are ranked by ep_morl_r -- the reward the agent actually optimized.

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

# Only the "old" env (zsceval/envs/overcooked/) implements the objective layer.
LAYOUT="random1"

cd $PYTHONPATH/zsceval/scripts/overcooked

# Offline sanity check on the objective vector before burning GPU hours.
python morl/check_morl_reward.py || exit 1

bash shell/train_morl.sh $LAYOUT

cd ..
python extract_models/extract_sp_models.py $LAYOUT overcooked morl ep_morl_r
