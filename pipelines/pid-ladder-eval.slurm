#!/bin/bash -l
#SBATCH -J ZSCEvalPIDLadderEval
#SBATCH -o /home-mscluster/vnaidoo/Coding/masters-research/logs/pid-ladder-eval-%j.log
#SBATCH -e /home-mscluster/vnaidoo/Coding/masters-research/logs/pid-ladder-eval-%j.err
#SBATCH --partition=bigbatch
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -t 12:00:00

# Closes out the partner-conditioning ladder: extract the rungs pid-ladder.sh
# trained, then score all three in one cross-play matrix against the held-out
# partners.
#
# The ladder is an ORACLE CEILING, not a ZSC method. Rung 2 and rung 3 show the
# actor its partner's identity, which a held-out partner does not have, so their
# number is not a zero-shot result. What the ladder buys is the decomposition:
# the gap from rung1 to rung3 is how much of the coordination shortfall is not
# knowing *who* the partner is (uncertainty), and whatever remains is inability
# to cooperate even when told (diversity).
#
# Usage: sbatch pid-ladder-eval.slurm [layout] [arm] [seeds] [rungs]
#
# Run pid-ladder.sh first. On random0 rungs 1 and 3 were trained for seeds 2-3
# only while rung 2 has 1-3, so top up before scoring or the rungs are compared
# on different seed counts:
#
#   bash pid-ladder.sh random0 bench_sp 1 1 rung1 rung3

if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
    echo "Environment variables loaded from wandb.env"
fi

if [ "$SHOULD_SOURCE_CONDA" = "true" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    echo "Skipping Conda source."
fi

conda activate zsceval

LAYOUT=${1:-random0}
ARM=${2:-bench_sp}
SEEDS=${3:-"1 2 3"}
RUNGS=${4:-"rung1 rung2 rung3"}
# The ladder's held-out set. `sp` only ever existed for random0; every other
# layout needs the bias agents, and the eval script folds the choice into the
# output filenames either way.
HELDOUT=${HELDOUT:-hsp}

cd $PYTHONPATH/zsceval/scripts

# The rungs are logged as fcp-S2-{arm}-{rung}, not one of the pipeline's fixed
# {algo}-S2-s{size} names, hence --exp.
for RUNG in $RUNGS; do
    echo "=== extracting ${ARM}-${RUNG}"
    python extract_models/extract_S2_models.py --layout $LAYOUT --env overcooked \
        --algorithm fcp --exp fcp-S2-${ARM}-${RUNG} || exit 1
done

# One matrix for the whole ladder: the rungs differ only in what the actor sees,
# so scoring them apart would put the control and the treatments in different
# tournaments against differently-sampled partners.
S2_ARMS=""
for RUNG in $RUNGS; do
    S2_ARMS="$S2_ARMS ${ARM}-${RUNG}"
done

cd $PYTHONPATH/../pipelines
HELDOUT=$HELDOUT bash morl-benchmark-eval.slurm "$LAYOUT" "$SEEDS" "$S2_ARMS"

echo "=========================================================="
echo "LADDER SCORED for $LAYOUT/$ARM at $(date)"
echo "Groups s2_pid_rung1..3 are in experiments/results/metrics_${LAYOUT}_s2_${HELDOUT}.json"
echo "=========================================================="
