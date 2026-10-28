#!/bin/bash -l
#SBATCH -J ZSCEvalHSPExtract
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/hsp-extract-%j.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/hsp-extract-%j.err
#SBATCH --partition=batch
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 04:00:00

# Extract HSP bias agents once their (one-job-per-seed) training has ended:
#   LAYOUTS="unident_s_cx pantry_cx" sbatch --dependency=afterany:<jobs> --export=ALL hsp-extract.slurm
# Only finished runs are extracted, so a seed that died is simply absent.

cd /home-mscluster/vnaidoo/Coding/masters-research/pipelines
if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
fi
if [ "$SHOULD_SOURCE_CONDA" = "true" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi
conda activate zsceval

EXP=${EXP:-hsp-s1}
cd $PYTHONPATH/zsceval/scripts
for L in ${LAYOUTS:-unident_s_cx pantry_cx}; do
    case "$L" in
        *_m|*_mx|*_cx) WB_PROJECT=Overcooked-new ;;
        *) WB_PROJECT=overcooked ;;
    esac
    python extract_models/extract_bias_agents_models.py $L "$WB_PROJECT" $EXP || exit 1
    echo "$L: $(ls $POLICY_POOL/$L/hsp/s1/$EXP/ 2>/dev/null | grep -c final) final w0/w1 checkpoints"
done
