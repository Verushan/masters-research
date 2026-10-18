#!/bin/bash
# Second submission of 18 Oct 2026 (run from the workstation; it pulls first):
#   ssh cluster 'bash -s' < pipelines/queue-2026-10-18b.sh
# 1. Re-run the specialist-arm seeds lost to the CPU-pinning crash (zsc-eval
#    _pin_worker), one seed per task, and re-chain their extraction.
# 2. Extract the stage-1 populations, then train the usual stage-2 arms on them:
#    hand-shaped ego (fcp-S2-usual-hand) and MORL ego (fcp-S2-usual-dishann).
# 3. Top up stage 1: bench_sp-cx-lr seeds 13-20 (competent held-out partners;
#    on pantry_cx 4 of seeds 7-12 never learned) and bench_morl_ann-cx-lr seeds
#    7-12 (MORL's stage-1 lead on unident_s_cx, +114, p=.14 at 6 seeds).
cd ~/Coding/masters-research && git fetch -q && B=~/untracked-backup-oct18b && mkdir -p $B && git diff --name-only --diff-filter=A HEAD origin/worktree-partner-effects | while read f; do [ -e "$f" ] && mkdir -p $B/$(dirname $f) && mv "$f" "$B/$f"; done; for f in $(git diff --name-only | grep -v "^zsc-eval$"); do mkdir -p $B/$(dirname $f); cp $f $B/$f; git checkout -- $f; done; (cd zsc-eval && git checkout -- zsceval/envs/overcooked_new/src/overcooked_ai_py/data/planners/ 2>/dev/null); git pull -q origin worktree-partner-effects && git submodule update -q --init zsc-eval
git log -1 --oneline; (cd zsc-eval && git log -1 --oneline)
grep -q _pin_worker zsc-eval/zsceval/envs/env_wrappers.py && [ -f pipelines/cx-extract.slurm ] && grep -q pantry_cx zsc-eval/zsceval/scripts/extract_models/extract_S2_models.py \
    || { echo "stale checkout; not submitting"; exit 1; }
cd pipelines
P="-p batch,bigbatch -c 6 -t 23:50:00 --exclude=mscluster74"
scancel 70402 70404 2>/dev/null
unset INCLUDE FAMILY SCRIPT_PARTNERS REPEAT EXP_TAG ARM LAYOUTS SEEDS DISH_FLAGS EXP_SUFFIX MORL_TEAM_TASK
specs() { local out=""; for a in "$@"; do out="$out ${a%%:*}:${a##*:}-${a##*:}"; done; echo $out; }

# 1. Specialist arms: the missing seeds.
declare -A MISSING=(
  [unident_s_cx]="hand:3 hand:4 hand:5 hand:8 hand:9 hand:10 dishann:2 dishann:3 dishann:4 dishann:5 dishann:7 dishann:8 dishann:9 dishann:10"
  [pantry_cx]="hand:3 hand:4 hand:5 hand:8 hand:9 hand:10 dishann:3 dishann:4 dishann:5"
)
for L in unident_s_cx pantry_cx; do
  S=$(specs ${MISSING[$L]}); n=$(echo $S | wc -w)
  T=$(LAYOUT=$L POP=scripted SPEC="$S" ARMS="hand dishann" sbatch --parsable --array=0-$((n-1)) $P --export=ALL fillin-train.slurm)
  E=$(LAYOUT=$L POP=scripted ARMS="hand dishann" MODE=extract sbatch --parsable --dependency=afterok:$T $P --export=ALL fillin-train.slurm)
  echo "$L specialists: train=$T ($n seeds) extract=$E"
done

# 2. Stage-1 extraction, then the usual stage-2 arms on bench_sp-cx-lr seeds 1-6.
X=$(sbatch --parsable cx-extract.slurm)
echo "stage-1 extraction: $X"
USUAL=$(specs hand:1 hand:2 hand:3 hand:4 hand:5 hand:6 hand:7 hand:8 hand:9 hand:10 dishann:1 dishann:2 dishann:3 dishann:4 dishann:5 dishann:6 dishann:7 dishann:8 dishann:9 dishann:10)
for L in unident_s_cx pantry_cx; do
  T=$(LAYOUT=$L POP=usual INCLUDE=$L/fcp/s2/train-bench_sp-cx-lr.yml SCRIPT_PARTNERS=none SPEC="$USUAL" ARMS="hand dishann" \
      sbatch --parsable --array=0-19 --dependency=afterok:$X $P --export=ALL fillin-train.slurm)
  E=$(LAYOUT=$L POP=usual ARMS="hand dishann" MODE=extract sbatch --parsable --dependency=afterok:$T $P --export=ALL fillin-train.slurm)
  echo "$L usual: train=$T extract=$E"
done

# 3. Stage-1 top-ups.
S1=$(LAYOUTS="unident_s_cx pantry_cx" SEEDS="13 14 15 16 17 18 19 20" DISH_FLAGS="--lr 2.5e-4 --critic_lr 2.5e-4" EXP_SUFFIX=-cx-lr \
     sbatch --parsable --array=0-15 $P --export=ALL dish-pilot.slurm)
S2=$(LAYOUTS="unident_s_cx pantry_cx" SEEDS="7 8 9 10 11 12" DISH_FLAGS="--lr 2.5e-4 --critic_lr 2.5e-4" EXP_SUFFIX=-cx-lr \
     ARM=bench_morl_ann MORL_TEAM_TASK=1 sbatch --parsable --array=0-11 $P --export=ALL dish-pilot.slurm)
echo "stage-1 top-ups: bench_sp 13-20=$S1 morl 7-12=$S2"
sleep 5; squeue -u $USER -h -o "%i %P %T %R" | head -30
