#!/bin/bash -l
#SBATCH -J ZSCEvalS2Replicate
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/s2-replicate-%j.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/s2-replicate-%j.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 12:00:00

# Recover the stage-2 runs the extractor was throwing away, then re-run cross-play.
#
# Nothing here trains. `extract_S2_models.py` writes `{seed}.pt`, so two finished
# runs of one seed collide, and the remedy in use was to tag one 'unused' in
# W&B. On unident_s that had been done to four runs whose configs are identical
# to the ones kept -- they are reruns, and PPO on GPU is not bit-deterministic,
# so each is a separate sample. Worse, the half that got tagged off was the
# stronger half of bench_sp (dropped mean 190.0 against a kept 162.8) while
# bench_morl kept nearly all of its runs, which puts a hand-picked per-arm
# subsample underneath the arm comparison the report leads with.
#
# --replicates keeps every finished run: the oldest of a seed keeps `{seed}.pt`
# so nothing already in the pool moves, and the rest become `{seed}r2.pt`. The
# union of labels is passed to the yml generator, which drops the ones a given
# arm does not have via --skip_missing.
#
# random0 has no duplicates. It is here because its cross_play_s2_hsp.yml was
# generated before seed 3 was extracted and so only ever listed _s1 and _s2 --
# the n=2 in metrics_random0_s2_hsp.json is that stale file, not missing runs.
#
# Usage: sbatch s2-replicate-eval.slurm

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

ARMS=${ARMS:-"bench_sp bench_morl bench_morl_ad"}
LAYOUTS=${LAYOUTS:-"unident_s random0"}
# Union of the labels any arm might carry; --skip_missing drops the rest.
LABELS=${LABELS:-"1 2 3 1r2 2r2 3r2"}

cd $PYTHONPATH/zsceval/scripts

for LAYOUT in $LAYOUTS; do
    for ARM in $ARMS; do
        echo "########## extract ${LAYOUT} ${ARM} ##########"
        python extract_models/extract_S2_models.py \
            --layout "$LAYOUT" --env Overcooked -a fcp \
            --exp "fcp-S2-${ARM}" --replicates || echo "  FAILED ${LAYOUT} ${ARM}"
    done
done

echo "=========================================================="
echo "stage-2 pool after extraction:"
for LAYOUT in $LAYOUTS; do
    for ARM in $ARMS; do
        D="$POLICY_POOL/${LAYOUT}/fcp/s2/fcp-S2-${ARM}"
        [ -d "$D" ] && echo "  ${LAYOUT}/${ARM}: $(ls $D | tr '\n' ' ')"
    done
done
echo "=========================================================="

cd /home-mscluster/vnaidoo/Coding/masters-research/pipelines
for LAYOUT in $LAYOUTS; do
    echo "########## cross-play ${LAYOUT} ##########"
    S2_ARM_SEEDS="$LABELS" HELDOUT=hsp \
        bash morl-benchmark-eval.slurm "$LAYOUT" "1 2 3" "$ARMS" \
        || echo "  FAILED cross-play ${LAYOUT}"
done
