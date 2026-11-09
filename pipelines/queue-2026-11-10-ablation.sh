#!/bin/bash
# Which MORL bonus makes MORL cook the valuable dish? (10 Nov 2026). Run from the workstation:
#   ssh cluster 'bash -s' < pipelines/queue-2026-11-10-ablation.sh
# Stage-1 annealed MORL (bench_morl_ann, `dishes` objectives, team-credited
# deliveries), unident_s_cx and pantry_cx, seeds 1-6, 5e6 steps, lr 2.5e-4, with
# one early (annealed) bonus removed. Weights are task_completion, dish_value,
# ingredient_prep, plating, coordination, recipe_quality; the base is 0,20,3,3,3,3.
#   norq      no recipe_quality (the bonus for keeping the best dish reachable)
#   nocoord   no coordination (counter handoffs the partner uses)
#   noprep    no ingredient_prep (any placement in a pot)
# The dish replays (experiments/dish_trajectory.py) then show which removal sends
# MORL back to onion soup on the open kitchen.
cd ~/Coding/masters-research && git fetch -q && B=~/untracked-backup-nov10 && mkdir -p $B && git diff --name-only --diff-filter=A HEAD origin/worktree-partner-effects | while read f; do [ -e "$f" ] && mkdir -p $B/$(dirname $f) && mv "$f" "$B/$f"; done; for f in $(git diff --name-only | grep -v "^zsc-eval$"); do mkdir -p $B/$(dirname $f); cp $f $B/$f; git checkout -- $f; done; (cd zsc-eval && git checkout -- zsceval/envs/overcooked_new/src/overcooked_ai_py/data/planners/ 2>/dev/null); git pull -q origin worktree-partner-effects && git submodule update -q --init zsc-eval
git log -1 --oneline
cd pipelines
P="-p batch -c 6 -t 23:50:00 --exclude=mscluster74"
unset ENT_FINAL DISH_FLAGS EXP_SUFFIX
export LAYOUTS="unident_s_cx pantry_cx" SEEDS="1 2 3 4 5 6" ARM=bench_morl_ann MORL_TEAM_TASK=1
LR="--lr 2.5e-4 --critic_lr 2.5e-4"
sub() { local j=$(env ANN_WEIGHTS=$2 DISH_FLAGS="$LR" EXP_SUFFIX=-cx-lr-$1 sbatch --parsable --array=0-11 $P --export=ALL dish-pilot.slurm); echo "$1 ($2): $j"; }
sub norq    0,20,3,3,3,0
sub nocoord 0,20,3,3,0,3
sub noprep  0,20,0,3,3,3
squeue -u $USER -h -r | wc -l
