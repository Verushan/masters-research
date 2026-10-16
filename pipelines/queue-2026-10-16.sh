#!/bin/bash
# One-off submission for 16 Oct 2026, run from the workstation (it pulls first):
#   ssh cluster 'bash -s' < pipelines/queue-2026-10-16.sh
# Stability diagnostic for the complex-dish kitchens. In the 15 Oct pilot, 4 of
# the 7 runs that learned collapsed in the last 1-2M steps (entropy back up to
# ~1.39, actor grad norm ~0.03, clip rate ~0.001). Three single-change variants
# of the hand-shaped baseline, same kitchens and seeds as the pilot, so each run
# has a direct before:
#   -cx-shape2  shaping fades over 1e7 steps (half strength at the end, never 0)
#   -cx-lr      learning rate halved (actor and critic 2.5e-4)
#   -cx-ent     final entropy coefficient 0.03 instead of 0.01
cd ~/Coding/masters-research && git fetch -q && B=~/untracked-backup-oct16 && mkdir -p $B && git diff --name-only --diff-filter=A HEAD origin/worktree-partner-effects | while read f; do [ -e "$f" ] && mkdir -p $B/$(dirname $f) && mv "$f" "$B/$f"; done; for f in $(git diff --name-only | grep -v "^zsc-eval$"); do mkdir -p $B/$(dirname $f); cp $f $B/$f; git checkout -- $f; done; (cd zsc-eval && git checkout -- zsceval/envs/overcooked_new/src/overcooked_ai_py/data/planners/ 2>/dev/null); git pull -q origin worktree-partner-effects && git submodule update -q --init zsc-eval
git log -1 --oneline; (cd zsc-eval && git log -1 --oneline)
grep -q ENT_FINAL zsc-eval/zsceval/scripts/overcooked/shell/train_morl_benchmark.sh || { echo "stale checkout: no ENT_FINAL; not submitting"; exit 1; }
cd pipelines
export LAYOUTS="unident_s_cx pantry_cx" SEEDS="1 2 3"
unset ARM MORL_TEAM_TASK ENT_FINAL DISH_FLAGS
J1=$(DISH_FLAGS="--reward_shaping_horizon 1e7" EXP_SUFFIX=-cx-shape2 sbatch --parsable --array=0-5 --exclude=mscluster74 --export=ALL dish-pilot.slurm)
J2=$(DISH_FLAGS="--lr 2.5e-4 --critic_lr 2.5e-4" EXP_SUFFIX=-cx-lr sbatch --parsable --array=0-5 --exclude=mscluster74 --export=ALL dish-pilot.slurm)
J3=$(ENT_FINAL=0.03 EXP_SUFFIX=-cx-ent sbatch --parsable --array=0-5 --exclude=mscluster74 --export=ALL dish-pilot.slurm)
echo "stability diagnostic: shape2=$J1 lr=$J2 ent=$J3"
sleep 5; squeue -u $USER -h -o "%i %j %T %R" | head -20
