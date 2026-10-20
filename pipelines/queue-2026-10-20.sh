#!/bin/bash
# Stage-2 stability test on pantry_cx before the 21 Oct meeting. Run from the workstation:
#   ssh cluster 'bash -s' < pipelines/queue-2026-10-20.sh
# The 5e6 specialist arms collapsed after ~2.5M steps (Finding 9/11). Two single
# changes to fcp-S2-scripted-hand at 5e6, seeds 1-5, a checkpoint every 50 updates
# (~0.24M steps) so a best checkpoint can be kept in future:
#   -5m-lrq    learning rate 1.25e-4 (a quarter of the default)
#   -5m-sched  schedules fixed at the 2e6 pace (S2_SCHED_STEPS=2e6), lr 2.5e-4
# then extraction (final) and pantry cross-play against the same held-out
# partners, with the 2e6 / 5e6 specialist arms and the usual agent as references.
cd ~/Coding/masters-research && git fetch -q && B=~/untracked-backup-oct20 && mkdir -p $B && git diff --name-only --diff-filter=A HEAD origin/worktree-partner-effects | while read f; do [ -e "$f" ] && mkdir -p $B/$(dirname $f) && mv "$f" "$B/$f"; done; for f in $(git diff --name-only | grep -v "^zsc-eval$"); do mkdir -p $B/$(dirname $f); cp $f $B/$f; git checkout -- $f; done; (cd zsc-eval && git checkout -- zsceval/envs/overcooked_new/src/overcooked_ai_py/data/planners/ 2>/dev/null); git pull -q origin worktree-partner-effects && git submodule update -q --init zsc-eval
git log -1 --oneline
grep -q S2_SCHED_STEPS zsc-eval/zsceval/scripts/overcooked/shell/train_morl_stage_2.sh || { echo "stale checkout; not submitting"; exit 1; }
cd pipelines
P="-p batch -c 6 -t 23:50:00 --exclude=mscluster74"
unset INCLUDE FAMILY SCRIPT_PARTNERS REPEAT EXP_TAG ARM LAYOUTS SEEDS S2_SCHED_STEPS S2_LR_FLAGS
SPEC="hand:1-1 hand:2-2 hand:3-3 hand:4-4 hand:5-5"
EX=""
run() {  # tag lr [sched]
  local T=$(env LAYOUT=pantry_cx POP=scripted STEPS=5e6 EXP_TAG=$1 SPEC="$SPEC" ARMS=hand S2_LR_FLAGS="--lr $2 --critic_lr $2 --save_interval 50" ${3:+S2_SCHED_STEPS=$3} \
      sbatch --parsable --array=0-4 $P --export=ALL fillin-train.slurm)
  local E=$(env LAYOUT=pantry_cx POP=scripted ARMS=hand EXP_TAG=$1 MODE=extract sbatch --parsable --dependency=afterok:$T $P --export=ALL fillin-train.slurm)
  EX="$EX:$E"; echo "$1: train=$T extract=$E"
}
run -5m-lrq 1.25e-4
run -5m-sched 2.5e-4 2e6
X=$(LAYOUT=pantry_cx ARMS="scripted-hand-5m-lrq scripted-hand-5m-sched scripted-hand scripted-hand-5m usual-hand" REF_S2="" S1_ARM="bench_sp-cx-lr" S1_SEEDS="1" \
    HELDOUT_ONLY=1 SKIP_EXTRACT=1 TAG=_sp_stability PARTNER_GROUP=heldout \
    HELDOUT_ARGS="--heldout sp --heldout_exp bench_sp-cx-lr --heldout_seeds 8 9 13 15 17 --heldout_tags mid final" \
    sbatch --parsable --dependency=afterok$EX $P --export=ALL fillin-crossplay.slurm)
echo "crossplay=$X"
squeue -u $USER -h -o "%i %j %T %R" | head
