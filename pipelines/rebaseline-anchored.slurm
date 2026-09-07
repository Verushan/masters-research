#!/bin/bash -l
#SBATCH -J ZSCEvalRebaseAnc
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/rebaseline-anc-%A_%a.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/rebaseline-anc-%A_%a.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 12:00:00

# Re-baseline the MORL arms under the anchored coordination objective, one array
# task per (arm, seed) so the cluster can run them in parallel.
#
# The arms whose reward is w . r_vec were trained against a farmable objective:
# on random0 `coordination` counts counter handoffs, the put and the pickup undo
# each other, and the best score reached while delivering nothing is 3.69x the
# median reached while delivering. Six seeds found the loop. Those runs measure
# reward hacking, and are tagged `superseded` in W&B rather than deleted, since
# they are also the evidence for that finding.
#
# bench_sp and bench_sparse are not re-run and are not in the spec below.
# Neither takes the objective vector as its reward -- bench_sparse is
# w = (20,0,0,0), task completion only, which is not farmable -- so the baseline
# is unaffected.
#
# EXP_SUFFIX keeps these out of the run set they replace: extract_sp_models
# filters on experiment_name, so reusing the original name would silently mix
# the new runs with the farmed ones.
#
# SPEC is a list of arm:seed pairs, one per array index. Defaults to the work
# left over when the local run was interrupted (bench_morl seeds 1-5 had already
# finished and are in W&B).
#
# Usage: sbatch --array=0-12 rebaseline-anchored.slurm

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

LAYOUT=${LAYOUT:-random0}
SPEC=${SPEC:-"bench_morl:6 \
bench_morl_ad:1 bench_morl_ad:2 bench_morl_ad:3 bench_morl_ad:4 bench_morl_ad:5 bench_morl_ad:6 \
bench_morl_div:1 bench_morl_div:2 bench_morl_div:3 bench_morl_div:4 bench_morl_div:5 bench_morl_div:6"}

read -r -a PAIRS <<< "$SPEC"
IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "${#PAIRS[@]}" ]; then
    echo "array index ${IDX} is past the ${#PAIRS[@]}-entry spec; nothing to do"
    exit 0
fi
PAIR=${PAIRS[$IDX]}
ARM=${PAIR%%:*}
SEED=${PAIR##*:}

export OBJECTIVES=${OBJECTIVES:-anchored}
export EXP_SUFFIX=${EXP_SUFFIX:--anc}
# The arms being replaced trained at 12 rollout threads, and .env sets a
# repo-wide value that would otherwise win here. Rollout threads are the PPO
# batch size, so a mismatch would make old and new incomparable -- and would
# also make this cluster half incomparable with the five seeds already run
# locally.
export ROLLOUT_THREADS=${REBASELINE_ROLLOUT_THREADS:-12}

echo "re-baseline task ${IDX}: layout ${LAYOUT}, arm ${ARM}, seed ${SEED}"
echo "objectives=${OBJECTIVES}, suffix=${EXP_SUFFIX}, rollout_threads=${ROLLOUT_THREADS}"

cd $PYTHONPATH/zsceval/scripts/overcooked
bash shell/train_morl_benchmark.sh "$LAYOUT" "$ARM" "$SEED" "$SEED"
