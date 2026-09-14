#!/bin/bash -l
#SBATCH -J ZSCEvalExtractHSP
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/extract-hsp-%A_%a.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/extract-hsp-%A_%a.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 2:00:00

# Extract already-trained HSP partners without retraining them.
#
# hsp-partners.slurm trains and then extracts in one job, with `|| exit 1` on
# the extraction. Jobs 50906 and 50952 trained all 16 partners each for
# unident_s_mx and unident_s_m, then died at that final line on a stale layout
# allow-list in extract_bias_agents_models.py. The runs are finished and synced;
# only the checkpoints never reached the pool. This is the extraction step on
# its own, so ~36 hours of training is not repeated for a one-line list.
#
# Usage: LAYOUTS="unident_s_mx unident_s_m" sbatch --array=0-1 --export=ALL extract-partners.slurm

cd /home-mscluster/vnaidoo/Coding/masters-research/pipelines

if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
fi
if [ "$SHOULD_SOURCE_CONDA" = "true" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi
conda activate zsceval

LAYOUTS=${LAYOUTS:-"unident_s_mx unident_s_m"}
read -r -a ARR <<< "$LAYOUTS"
LAYOUT=${ARR[${SLURM_ARRAY_TASK_ID:-0}]}
EXP=${EXP:-hsp-s1}

cd $PYTHONPATH/zsceval/scripts
echo "extracting ${EXP} partners for ${LAYOUT}"
# train_sp.py sends new-env runs to `{env}-new`, so the multi-recipe layouts
# live in a different W&B project. Extracting them from Overcooked finds
# nothing (job 54488: "num of runs: 0", exit 0, empty pool).
case "$LAYOUT" in
    *_m|*_mx) WB_PROJECT=Overcooked-new ;;
    *) WB_PROJECT=overcooked ;;
esac
python extract_models/extract_bias_agents_models.py "$LAYOUT" "$WB_PROJECT" "$EXP" || exit 1

D="$POLICY_POOL/${LAYOUT}/hsp/s1/${EXP}"
N=$(ls $D/*_final_w0_actor.pt 2>/dev/null | wc -l)
echo "pool now holds ${N} final w0 partners for ${LAYOUT}"
# An empty pool is a failure, whatever the extractor's exit code said.
[ "$N" -gt 0 ] || exit 1
