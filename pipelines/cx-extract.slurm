#!/bin/bash -l
#SBATCH -J ZSCEvalCxExtract
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/cx-extract-%j.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/cx-extract-%j.err
#SBATCH --partition=batch,bigbatch
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 04:00:00

# Phase 4: extract the complex-dish stage-1 agents and write the stage-2
# population ymls built from them.
#
# Every finished run of each experiment is extracted (init/mid/final by
# ep_sparse_r, which the annealed MORL reward also ends on), so this can be
# re-run as top-up seeds finish. The population is seeds POP_SEEDS of the
# hand-shaped arm; the rest are candidates for the held-out set.
#
#   sbatch cx-extract.slurm
#   LAYOUTS=pantry_cx sbatch --export=ALL cx-extract.slurm

cd /home-mscluster/vnaidoo/Coding/masters-research/pipelines
if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
fi
if [ "$SHOULD_SOURCE_CONDA" = "true" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi
conda activate zsceval

LAYOUTS=${LAYOUTS:-"unident_s_cx pantry_cx"}
EXPS=${EXPS:-"bench_sp-cx-lr bench_morl_ann-cx-lr"}
POP_SEEDS=${POP_SEEDS:-"1 2 3 4 5 6"}

cd $PYTHONPATH/zsceval/scripts
for L in $LAYOUTS; do
    python prep/store_policy_config.py $L || exit 1
    for E in $EXPS; do
        python extract_models/extract_sp_models.py $L Overcooked-new $E || exit 1
        echo "$L $E: $(ls $POLICY_POOL/$L/fcp/s1/$E/ 2>/dev/null | wc -l) checkpoints"
    done
    python prep/gen_arm_S2_yml.py $L --arm bench_sp-cx-lr --arm_seeds $POP_SEEDS || exit 1
done
echo "cx extraction done at $(date)"
