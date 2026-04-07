#!/bin/bash -l
#SBATCH -p bigbatch
#SBATCH -J ZSCEval
#SBATCH -o /home-mscluster/vnaidoo/Coding/zsceval/output.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/zsceval/output.err
#SBATCH -n 1                   # Number of tasks (processes)
#SBATCH -c 16                   # Number of CPU cores per task
#SBATCH -t 72:00:00            # Max runtime (hh:mm:ss)

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Environment variables loaded from wandb.env"
fi

LAYOUT="random0"
POPULATION_SIZE=3

source ~/anaconda3/etc/profile.d/conda.sh
conda activate zsceval # Activate conda environment

# Set project path for python modules
export PYTHONPATH=/home-mscluster/vnaidoo/Coding/zsceval/ZSC-Eval/

cd /home-mscluster/vnaidoo/Coding/zsceval/ZSC-Eval/zsceval/scripts/overcooked

bash shell/train_sp.sh $LAYOUT

cd ..
python extract_models/extract_sp_models.py $LAYOUT overcooked

# Stage 2
python prep/gen_S2_yml.py $LAYOUT fcp

cd overcooked
bash shell/train_fcp_stage_2.sh $LAYOUT $POPULATION_SIZE

cd ..
python extract_models/extract_S2_models.py --layout $LAYOUT --env overcooked --algorithm fcp
