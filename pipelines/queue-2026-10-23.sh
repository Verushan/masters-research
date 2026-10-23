#!/bin/bash
# The next round on the complex-dish kitchens (23 Oct 2026), run from the workstation:
#   ssh cluster 'bash -s' < pipelines/queue-2026-10-23.sh
# 1. HSP partners (ZSC-Eval's own held-out set) on unident_s_cx and pantry_cx:
#    16 bias agents each (select_hsp_seeds, k=16), 5e6 steps, lr 2.5e-4; their
#    biases include potting penalties and dish-size bonuses, so they cook a
#    variety of dishes where the self-play held-out set all cooked one.
#    Then extraction, and a best response per partner (BR-Prox denominators).
# 2. Stage 2 again, stable: 4 arms x 10 seeds per kitchen at 5e6 with schedules
#    at the 2e6 pace (S2_SCHED_STEPS) and a checkpoint every 50 updates, each run
#    keeping its peak on the training-partner curve (S2_CHECKPOINT=best).
#    Experiments fcp-S2-{partial,scripted,usual}-{hand,dishann}-v2.
# 3. Scoring once both are done: cross-play against the HSP partners (and the
#    self-play held-out set, for continuity) and the fill-in suite.
cd ~/Coding/masters-research && git fetch -q && B=~/untracked-backup-oct23 && mkdir -p $B && git diff --name-only --diff-filter=A HEAD origin/worktree-partner-effects | while read f; do [ -e "$f" ] && mkdir -p $B/$(dirname $f) && mv "$f" "$B/$f"; done; for f in $(git diff --name-only | grep -v "^zsc-eval$"); do mkdir -p $B/$(dirname $f); cp $f $B/$f; git checkout -- $f; done; (cd zsc-eval && git checkout -- zsceval/envs/overcooked_new/src/overcooked_ai_py/data/planners/ 2>/dev/null); git pull -q origin worktree-partner-effects && git submodule update -q --init zsc-eval
git log -1 --oneline
grep -q NO_EXTRACT pipelines/hsp-partners.slurm && grep -q '"best"' zsc-eval/zsceval/scripts/extract_models/extract_S2_models.py || { echo "stale checkout; not submitting"; exit 1; }
cd pipelines
PB="-p batch -c 6 -t 23:50:00 --exclude=mscluster74"
PBB="-p bigbatch -c 14 -t 48:00:00 --exclude=mscluster74"
unset INCLUDE FAMILY SCRIPT_PARTNERS REPEAT EXP_TAG ARM LAYOUTS SEEDS S2_SCHED_STEPS S2_LR_FLAGS S2_CHECKPOINT STEPS
LR="--lr 2.5e-4 --critic_lr 2.5e-4"

# 1. HSP partners: two seeds per bigbatch job, extraction once all have ended.
HSP_SEEDS="1 5 7 11 21 25 26 30 40 44 45 50 51 65 66 71"
declare -A HSPX BRJ
for L in unident_s_cx pantry_cx; do
  J=""; set -- $HSP_SEEDS
  while [ $# -gt 0 ]; do
    j=$(LAYOUTS=$L SEEDS="$1 $2" STEPS=5e6 BIAS_FLAGS="$LR" NO_EXTRACT=1 CONCURRENCY=2 \
        sbatch --parsable --array=0 $PBB --export=ALL hsp-partners.slurm); J="$J:$j"; shift 2
  done
  X=$(sbatch --parsable -p batch -c 2 -t 04:00:00 --dependency=afterany$J --wrap "cd ~/Coding/masters-research && export \$(grep -v '^#' .env | xargs) && source ~/anaconda3/etc/profile.d/conda.sh && conda activate zsceval && cd zsc-eval/zsceval/scripts && python extract_models/extract_bias_agents_models.py $L Overcooked-new hsp-s1 && ls \$POLICY_POOL/$L/hsp/s1/hsp-s1/")
  HSPX[$L]=$X
  BRJ[$L]=$(LAYOUTS=$L CONCURRENCY=3 sbatch --parsable --array=0 --dependency=afterok:$X $PB --export=ALL br-partners.slurm)
  echo "$L HSP: train$J extract=$X br=${BRJ[$L]}"
done

# 2. Stage 2, stable.
SPEC=""; for a in hand dishann; do for s in 1 3 5 7 9; do SPEC="$SPEC $a:$s-$((s+1))"; done; done
declare -A S2X
s2() {  # layout pop [env...]
  local L=$1 POPN=$2; shift 2
  local T=$(env LAYOUT=$L POP=$POPN STEPS=5e6 EXP_TAG=-v2 SPEC="$SPEC" ARMS="hand dishann" S2_SCHED_STEPS=2e6 S2_LR_FLAGS="$LR --save_interval 50" "$@" \
      sbatch --parsable --array=0-9 $PB --export=ALL fillin-train.slurm)
  local E=$(env LAYOUT=$L POP=$POPN EXP_TAG=-v2 ARMS="hand dishann" MODE=extract S2_CHECKPOINT=best "$@" \
      sbatch --parsable --dependency=afterok:$T $PB --export=ALL fillin-train.slurm)
  S2X[$L]="${S2X[$L]}:$E"; echo "$L $POPN v2: train=$T extract=$E"
}
s2 unident_s_cx partial SCRIPT_PARTNERS="potter server idle onion_potter tomato_potter"
s2 pantry_cx scripted
s2 unident_s_cx usual INCLUDE=unident_s_cx/fcp/s2/train-bench_sp-cx-lr.yml SCRIPT_PARTNERS=none
s2 pantry_cx usual INCLUDE=pantry_cx/fcp/s2/train-bench_sp-cx-lr.yml SCRIPT_PARTNERS=none

# 3. Scoring.
declare -A ARMS2=([unident_s_cx]="partial-hand-v2 partial-dishann-v2 usual-hand-v2 usual-dishann-v2"
                  [pantry_cx]="scripted-hand-v2 scripted-dishann-v2 usual-hand-v2 usual-dishann-v2")
declare -A EXPS2=([unident_s_cx]="fcp-S2-partial-hand-v2 fcp-S2-partial-dishann-v2 fcp-S2-usual-hand-v2 fcp-S2-usual-dishann-v2"
                  [pantry_cx]="fcp-S2-scripted-hand-v2 fcp-S2-scripted-dishann-v2 fcp-S2-usual-hand-v2 fcp-S2-usual-dishann-v2")
declare -A HO=([unident_s_cx]="7 9 10 12 15 16 17 19 20" [pantry_cx]="8 9 13 15 17")
for L in unident_s_cx pantry_cx; do
  D="--dependency=afterok${S2X[$L]}:${HSPX[$L]}"
  H=$(env LAYOUT=$L ARMS="${ARMS2[$L]}" REF_S2="" S1_ARM="bench_sp-cx-lr bench_morl_ann-cx-lr" S1_SEEDS="1 2 3 4 5 6" HELDOUT_ONLY=1 SKIP_EXTRACT=1 \
      TAG=_hsp_v2 PARTNER_GROUP=heldout_hsp HELDOUT_ARGS="--heldout hsp --hsp_partners auto" sbatch --parsable $D $PB --export=ALL fillin-crossplay.slurm)
  S=$(env LAYOUT=$L ARMS="${ARMS2[$L]}" REF_S2="" S1_ARM="bench_sp-cx-lr bench_morl_ann-cx-lr" S1_SEEDS="1 2 3 4 5 6" HELDOUT_ONLY=1 SKIP_EXTRACT=1 \
      TAG=_sp_v2 PARTNER_GROUP=heldout HELDOUT_ARGS="--heldout sp --heldout_exp bench_sp-cx-lr --heldout_seeds ${HO[$L]} --heldout_tags mid final" \
      sbatch --parsable --dependency=afterok${S2X[$L]} $PB --export=ALL fillin-crossplay.slurm)
  F=$(env OUT=../experiments/results/fillin/$L-v2 MANIFEST_ARGS="--s1_arms --s2_exps ${EXPS2[$L]}" BASE=s2_usual-hand-v2 \
      METRICS=fillin_metrics_${L}_v2.json REFERENCE_SEEDS="1 2 3" sbatch --parsable --dependency=afterok${S2X[$L]} $PB --export=ALL fillin-suite.sh $L)
  echo "$L scoring: hsp-crossplay=$H sp-crossplay=$S fillin=$F"
done
squeue -u $USER -h | wc -l; squeue -u $USER -h -r | wc -l
