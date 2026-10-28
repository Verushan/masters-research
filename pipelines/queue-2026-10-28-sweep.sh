#!/bin/bash
# Hyperparameter sweep for the complex-dish kitchens (28 Oct 2026). Run from the workstation:
#   ssh cluster 'bash -s' < pipelines/queue-2026-10-28-sweep.sh
# Stage-1 hand-shaped self-play (bench_sp), unident_s_cx and pantry_cx, seeds 1-6
# (the same seeds as bench_sp-cx-lr, the base), 5e6 steps, lr 2.5e-4, one change each:
#   ppo5       5 PPO epochs per batch instead of 15 (smaller policy change per batch)
#   mb4        4 minibatches instead of 2
#   lrdecay    learning rate decays linearly to 0
#   clip01     PPO clip 0.1 instead of 0.2
#   shapekeep  the shaping reward never fades (a recovery signal stays)
#   ent0       final entropy coefficient 0.001 instead of 0.01
# Read against bench_sp-cx-lr: runs that learned, runs that collapsed, final return.
cd ~/Coding/masters-research && git fetch -q && B=~/untracked-backup-oct28b && mkdir -p $B && git diff --name-only --diff-filter=A HEAD origin/worktree-partner-effects | while read f; do [ -e "$f" ] && mkdir -p $B/$(dirname $f) && mv "$f" "$B/$f"; done; for f in $(git diff --name-only | grep -v "^zsc-eval$"); do mkdir -p $B/$(dirname $f); cp $f $B/$f; git checkout -- $f; done; (cd zsc-eval && git checkout -- zsceval/envs/overcooked_new/src/overcooked_ai_py/data/planners/ 2>/dev/null); git pull -q origin worktree-partner-effects && git submodule update -q --init zsc-eval
git log -1 --oneline
cd pipelines
P="-p batch -c 6 -t 23:50:00 --exclude=mscluster74"
unset ARM MORL_TEAM_TASK ENT_FINAL DISH_FLAGS EXP_SUFFIX
export LAYOUTS="unident_s_cx pantry_cx" SEEDS="1 2 3 4 5 6"
LR="--lr 2.5e-4 --critic_lr 2.5e-4"
sub() { local tag=$1; shift; local j=$(env "$@" EXP_SUFFIX=-cx-lr-$tag sbatch --parsable --array=0-11 $P --export=ALL dish-pilot.slurm); echo "$tag: $j"; }
sub ppo5      DISH_FLAGS="$LR --ppo_epoch 5"
sub mb4       DISH_FLAGS="$LR --num_mini_batch 4"
sub lrdecay   DISH_FLAGS="$LR --use_linear_lr_decay"
sub clip01    DISH_FLAGS="$LR --clip_param 0.1"
sub shapekeep DISH_FLAGS="$LR --reward_shaping_horizon 1e12"
sub ent0      DISH_FLAGS="$LR" ENT_FINAL=0.001
squeue -u $USER -h | wc -l
