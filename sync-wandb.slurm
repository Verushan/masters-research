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

if [ "$SHOULD_SOURCE_CONDA" = "true" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate zsceval
else
    echo "Skipping Conda source."
fi

SEARCH_DIR="$HOME/ZSC"

echo "Searching for log files in $SEARCH_DIR..."

find "$SEARCH_DIR" -name "*.wandb" | xargs -I {} dirname {} | sort -u | while read -r run_dir; do
    echo "------------------------------------------------"
    echo "Found run data in: $run_dir"

    # Run the sync command
    # --mark-synced prevents re-uploading the same data later
    wandb sync --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" --mark-synced "$run_dir"

    if [ $? -eq 0 ]; then
        echo "Successfully processed $run_dir"
    else
        echo "Error: Failed to sync $run_dir"
    fi
done

echo "------------------------------------------------"
echo "Batch sync process complete."