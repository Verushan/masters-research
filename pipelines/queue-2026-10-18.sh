#!/bin/bash
# One-off submission for 18 Oct 2026, run from the workstation (it pulls first):
#   ssh cluster 'bash -s' < pipelines/queue-2026-10-18.sh
# Phase 4, the specialist arms on the complex-dish kitchens. They train against
# scripted cooks only, so they do not wait for the stage-1 populations (70382,
# 70383). Per kitchen: hand-shaped ego (A) and MORL ego (dishann: annealed
# `dishes` objectives, team-credited deliveries), 10 seeds each, swaps at step
# 200, lr 2.5e-4, 2e6 steps; experiments fcp-S2-scripted-{hand,dishann}.
cd ~/Coding/masters-research && git fetch -q && B=~/untracked-backup-oct18 && mkdir -p $B && git diff --name-only --diff-filter=A HEAD origin/worktree-partner-effects | while read f; do [ -e "$f" ] && mkdir -p $B/$(dirname $f) && mv "$f" "$B/$f"; done; for f in $(git diff --name-only | grep -v "^zsc-eval$"); do mkdir -p $B/$(dirname $f); cp $f $B/$f; git checkout -- $f; done; (cd zsc-eval && git checkout -- zsceval/envs/overcooked_new/src/overcooked_ai_py/data/planners/ 2>/dev/null); git pull -q origin worktree-partner-effects && git submodule update -q --init zsc-eval
git log -1 --oneline; (cd zsc-eval && git log -1 --oneline)
grep -q dishann pipelines/fillin-train.slurm || { echo "stale checkout: no dishann arm; not submitting"; exit 1; }
# Policy configs the population ymls name (layout-shaped; seconds to build).
source ~/anaconda3/etc/profile.d/conda.sh; conda activate zsceval; export $(grep -v '^#' .env | xargs)
(cd zsc-eval/zsceval/scripts && python prep/store_policy_config.py unident_s_cx pantry_cx 2>&1 | grep -c SUCCESS) || exit 1
cd pipelines
unset INCLUDE FAMILY SCRIPT_PARTNERS REPEAT EXP_TAG ARM
for L in unident_s_cx pantry_cx; do
  T=$(LAYOUT=$L POP=scripted SPEC="hand:1-5 hand:6-10 dishann:1-5 dishann:6-10" ARMS="hand dishann" \
      sbatch --parsable --array=0-3 --exclude=mscluster74 --export=ALL fillin-train.slurm)
  E=$(LAYOUT=$L POP=scripted ARMS="hand dishann" MODE=extract \
      sbatch --parsable --dependency=afterok:$T --exclude=mscluster74 --export=ALL fillin-train.slurm)
  echo "$L: train=$T extract=$E"
done
# Stage-1 population health: tasks that died at startup need a single-seed resubmit.
sacct -j 70382,70383 -X -n -o JobID%12,State%10 | awk '{print $2}' | sort | uniq -c
sleep 5; squeue -u $USER -h -o "%i %j %T %R" | head -20
