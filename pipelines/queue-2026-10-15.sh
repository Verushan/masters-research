#!/bin/bash
# One-off submission for 15 Oct 2026, run from the workstation (it pulls first):
#   ssh cluster 'bash -s' < pipelines/queue-2026-10-15.sh
# Phase 1 (no-swap training, swap testing; preregistration-2026-10-15-swaps.md):
#   unident_s noann2ns x10, random1 handns + noann2ns x10 -> extract -> zero-shot, fill-in, swap cross-play.
# Phase 2a/3 learnability pilot: dish-pilot.slurm, bench_sp and bench_morl_ann on three _cx kitchens.
# The pull backs up colliding untracked/modified files to ~/untracked-backup-oct15 first.
cd ~/Coding/masters-research && git fetch -q && B=~/untracked-backup-oct15 && mkdir -p $B && git diff --name-only --diff-filter=A HEAD origin/worktree-partner-effects | while read f; do [ -e "$f" ] && mkdir -p $B/$(dirname $f) && mv "$f" "$B/$f"; done; for f in $(git diff --name-only | grep -v "^zsc-eval$"); do mkdir -p $B/$(dirname $f); cp $f $B/$f; git checkout -- $f; done; (cd zsc-eval && git checkout -- zsceval/envs/overcooked_new/src/overcooked_ai_py/data/planners/ 2>/dev/null); git pull -q origin worktree-partner-effects && git submodule update -q --init zsc-eval
git log -1 --oneline; (cd zsc-eval && git log -1 --oneline)
cd pipelines
RELAXX="--morl_neglect_prior 0.5 --morl_neglect_halflife 10"
chain() {  # layout spec arms
  export LAYOUT=$1 POP=scripted ARMS="$3"; unset INCLUDE FAMILY SCRIPT_PARTNERS REPEAT EXP_TAG
  n=$(echo $2 | wc -w)
  T=$(SPEC="$2" sbatch --parsable --array=0-$((n-1)) --exclude=mscluster74 --export=ALL fillin-train.slurm)
  E=$(MODE=extract sbatch --parsable --dependency=afterok:$T --exclude=mscluster74 --export=ALL fillin-train.slurm)
  XARMS=$(for a in $3; do echo -n "scripted-$a "; done)
  X=$(ARMS="$XARMS" HELDOUT_ONLY=1 SKIP_EXTRACT=1 TAG=_hsp_noswap NEGLECT_EXTRA="$RELAXX" sbatch --parsable -t 16:00:00 --dependency=afterok:$E --exclude=mscluster74 --export=ALL fillin-crossplay.slurm)
  F=$(OUT=../experiments/results/fillin/$1-noswap MANIFEST_ARGS="--s2_exps fcp-S2-scripted-hand fcp-S2-scripted-handns fcp-S2-scripted-noann2 fcp-S2-scripted-noann2ns" BASE=s2_scripted-handns METRICS=fillin_metrics_$1_noswap.json sbatch --parsable --dependency=afterok:$E --exclude=mscluster74 --export=ALL fillin-suite.sh $1)
  SW=$(ARMS="scripted-hand scripted-handns scripted-noann2 scripted-noann2ns" HELDOUT_ONLY=1 SKIP_EXTRACT=1 TAG=_${4}_swaps SWAP_STEP=200 NEGLECT_EXTRA="$RELAXX" HELDOUT_ARGS="$5" PARTNER_GROUP="$6" sbatch --parsable -t 16:00:00 --dependency=afterok:$E --exclude=mscluster74 --export=ALL fillin-crossplay.slurm)
  echo "$1: train=$T extract=$E crossplay=$X fillin=$F swaps=$SW"
}
chain unident_s "noann2ns:1-5 noann2ns:6-10" "noann2ns" hsp "--heldout hsp --hsp_partners auto" heldout_hsp
chain random1 "handns:1-5 handns:6-10 noann2ns:1-5 noann2ns:6-10" "handns noann2ns" sp "--heldout sp --heldout_exp bench_sp --heldout_seeds 7 8 9 10 11 12 --heldout_tags mid final" heldout
unset LAYOUT ARMS ARM MORL_TEAM_TASK
D1=$(sbatch --parsable --array=0-8 --exclude=mscluster74 dish-pilot.slurm)
D2=$(ARM=bench_morl_ann MORL_TEAM_TASK=1 sbatch --parsable --array=0-8 --exclude=mscluster74 --export=ALL dish-pilot.slurm)
echo "dish pilot: bench_sp=$D1 bench_morl_ann=$D2"
sleep 5; squeue -u $USER -h -o "%i %j %T %R" | head -20
