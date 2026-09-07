#!/bin/bash -l
#SBATCH -J ZSCEvalRebaseEval
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/rebaseline-eval-%j.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/rebaseline-eval-%j.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 10:00:00

# Score the anchored re-baseline: extract its checkpoints, then cross-play them
# against the same held-out partners as everything else.
#
# rebaseline-anchored.slurm only trains. Without this the anchored arms have
# stage-1 self-play returns and no ZSC number at all, and self-play return is
# precisely the thing that does not answer the question -- a MORL agent
# coordinating with itself was never in doubt.
#
# bench_sp and bench_sparse are in the pool unchanged. Neither takes the
# objective vector as its reward, so neither was re-trained, and bench_sp is the
# baseline the whole comparison is against.
#
# The MORL arms are ranked on ep_morl_r for checkpoint selection, as the original
# benchmark did -- their own reward is what they optimised. That metric is not
# comparable *across* bench_morl_div's members, since each optimises a different
# w, but selection only ever compares a run against itself.
#
# Usage: sbatch rebaseline-eval.slurm [layout]

cd /home-mscluster/vnaidoo/Coding/masters-research/pipelines

if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
    echo "Environment variables loaded from .env"
fi

if [ "$SHOULD_SOURCE_CONDA" = "true" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    echo "Skipping Conda source."
fi

conda activate zsceval

LAYOUT=${1:-random0}
ANC_ARMS=${ANC_ARMS:-"bench_morl-anc bench_morl_ad-anc bench_morl_div-anc"}
BASE_ARMS=${BASE_ARMS:-"bench_sp bench_sparse"}
SEEDS=${SEEDS:-"1 2 3 4 5 6"}

cd $PYTHONPATH/zsceval/scripts
for ARM in $ANC_ARMS; do
    echo "########## extract ${ARM} ##########"
    python extract_models/extract_sp_models.py "$LAYOUT" overcooked "$ARM" ep_morl_r \
        || echo "  extraction failed for ${ARM}"
done

echo "=========================================================="
echo "stage-1 pool after extraction:"
for ARM in $ANC_ARMS $BASE_ARMS; do
    D="$POLICY_POOL/${LAYOUT}/fcp/s1/${ARM}"
    [ -d "$D" ] && echo "  ${ARM}: $(ls $D 2>/dev/null | wc -l) files"
done
echo "=========================================================="

cd /home-mscluster/vnaidoo/Coding/masters-research/pipelines
ARMS="$BASE_ARMS $ANC_ARMS" HELDOUT=hsp \
    bash morl-benchmark-eval.slurm "$LAYOUT" "$SEEDS"

echo "=========================================================="
echo "rebaseline eval done for ${LAYOUT} at $(date)"
echo "=========================================================="
