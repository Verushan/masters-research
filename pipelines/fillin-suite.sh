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
$PY ../experiments/fillin_manifest.py "$LAYOUT" > "$MANIFEST" || exit 1
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

N=$(ls "$OUT"/*.json.gz 2>/dev/null | wc -l)
echo "${N} of $(wc -l < "$MANIFEST") agents recorded"
$PY ../experiments/analyze_fillin.py "$OUT"/*.json.gz \
    --out "../experiments/results/fillin_metrics_${LAYOUT}.json"
