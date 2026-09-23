#!/bin/bash
#SBATCH -J ZSCEvalFillIn
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/fillin-suite-%j.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/fillin-suite-%j.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 4:00:00

# Fill-In Evaluation Suite, step 5: every trained agent on a layout, through
# the fill-in harness, in parallel, then the analysis.
#
#   bash fillin-suite.sh unident_s                    # locally
#   sbatch fillin-suite.sh unident_s                  # on the cluster
#   FILLIN_POOL=/path/to/pool PY=/path/to/python bash fillin-suite.sh unident_s
#
# FILLIN_POOL overrides POLICY_POOL: .env exports POLICY_POOL and is loaded
# first, so a plain POLICY_POOL=... on the command line would be overwritten.
#
# Step 8 scores only the Step 7 arms, in a directory of its own, against the
# hand-shaped arm, with the scripted ceiling re-run on the same schedules:
#
#   OUT=../experiments/results/fillin/unident_s-step8 \
#   MANIFEST_ARGS="--s1_arms --s2_exps fcp-S2-scripted-hand fcp-S2-scripted-neglect fcp-S2-scripted-noann" \
#   BASE=s2_scripted-hand METRICS=fillin_metrics_unident_s_step8.json REFERENCE_SEEDS="1 2 3 4 5 6" \
#   bash fillin-suite.sh unident_s

HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$HERE" || exit 1
if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
fi
if [ "$SHOULD_SOURCE_CONDA" = "true" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate zsceval
fi
export POLICY_POOL=${FILLIN_POOL:-$POLICY_POOL}
PY=${PY:-python}
# One thread per process: JOBS torch processes each spawning a thread per core
# oversubscribe the machine and run slower than fewer processes would.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
LAYOUT=${1:-unident_s}
EPISODES=${EPISODES:-5}
JOBS=${JOBS:-${SLURM_CPUS_PER_TASK:-8}}
OUT=${OUT:-../experiments/results/fillin/${LAYOUT}}
mkdir -p "$OUT"

MANIFEST="$OUT/manifest.tsv"
$PY ../experiments/fillin_manifest.py "$LAYOUT" ${MANIFEST_ARGS} > "$MANIFEST" || exit 1
echo "$(wc -l < "$MANIFEST") agents on ${LAYOUT}, ${EPISODES} episodes per schedule per seat, ${JOBS} in parallel"

# One process per agent; each writes its own file, so a failure loses one agent.
run_one() {
    IFS=$'\t' read -r name actor config flags <<< "$1"
    [ -f "$OUT/${name}.json.gz" ] && return 0
    $PY ../experiments/fillin_eval.py --layout "$LAYOUT" --agent "${name}=${actor}" \
        --config "$config" --env_flags "$flags" --episodes "$EPISODES" \
        --out "$OUT/${name}.json.gz" > "$OUT/${name}.log" 2>&1 \
        && echo "  done ${name}" || echo "  FAILED ${name} (see $OUT/${name}.log)"
}
export -f run_one
export PY LAYOUT EPISODES OUT
tr '\n' '\0' < "$MANIFEST" | xargs -0 -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}

# The scripted ceiling (Step 6), on whatever schedules the harness now has.
for s in ${REFERENCE_SEEDS}; do
    for mode in oracle generalist; do
        f="$OUT/ref_${mode}_s${s}.json.gz"
        [ -f "$f" ] || $PY ../experiments/fillin_reference.py --layout "$LAYOUT" --mode $mode \
            --seed $((s - 1)) --episodes "$EPISODES" --out "$f" > "$OUT/ref_${mode}_s${s}.log" 2>&1 \
            || echo "  FAILED ref_${mode}_s${s}"
    done
done

N=$(ls "$OUT"/*.json.gz 2>/dev/null | wc -l)
echo "${N} of $(wc -l < "$MANIFEST") agents recorded"
$PY ../experiments/analyze_fillin.py "$OUT"/*.json.gz --base "${BASE:-bench_sp}" \
    --out "../experiments/results/${METRICS:-fillin_metrics_${LAYOUT}.json}"
