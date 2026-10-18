#!/bin/bash
# Phase 4 at the stage-1 budget (18 Oct 2026). Run from the workstation:
#   ssh cluster 'bash -s' < pipelines/queue-2026-10-18c.sh
# The 2e6 stage-2 agents were undertrained (stage 1 on these kitchens only
# starts learning at 1.5-3M steps), and on unident_s_cx the full scripted cooks
# ran the kitchen alone, so the ego free-rode (training return flat at ~110 from
# step 0). Everything reruns at 5e6 with EXP_TAG -5m; on unident_s_cx the
# specialists are partial ones only (POP=partial: potter, server, idle, onion-
# and tomato-only potters), the dials and generalist staying fill-in test
# partners. pantry_cx keeps its 7 (its scripts cannot cook alone).
# Then: 5e6 best responses, cross-play (both held-out groups on unident_s_cx),
# and the fill-in suite, chained on the extractions.
cd ~/Coding/masters-research && git fetch -q && B=~/untracked-backup-oct18d && mkdir -p $B && git diff --name-only --diff-filter=A HEAD origin/worktree-partner-effects | while read f; do [ -e "$f" ] && mkdir -p $B/$(dirname $f) && mv "$f" "$B/$f"; done; for f in $(git diff --name-only | grep -v "^zsc-eval$"); do mkdir -p $B/$(dirname $f); cp $f $B/$f; git checkout -- $f; done; (cd zsc-eval && git checkout -- zsceval/envs/overcooked_new/src/overcooked_ai_py/data/planners/ 2>/dev/null); git pull -q origin worktree-partner-effects && git submodule update -q --init zsc-eval
git log -1 --oneline
grep -q BR_LOGDIR pipelines/br-partners.slurm || { echo "stale checkout; not submitting"; exit 1; }
cd pipelines
P="-p batch -c 6 -t 23:50:00 --exclude=mscluster74"
unset INCLUDE FAMILY SCRIPT_PARTNERS REPEAT EXP_TAG ARM LAYOUTS SEEDS DISH_FLAGS EXP_SUFFIX MORL_TEAM_TASK
ALL=""; for a in hand dishann; do for s in $(seq 1 10); do ALL="$ALL $a:$s-$s"; done; done
export STEPS=5e6 EXP_TAG=-5m
declare -A EXTRACT
train() {  # layout pop [extra env assignments...]
  local L=$1 POPN=$2; shift 2
  local T=$(env LAYOUT=$L POP=$POPN SPEC="$ALL" ARMS="hand dishann" "$@" sbatch --parsable --array=0-19 $P --export=ALL fillin-train.slurm)
  local E=$(env LAYOUT=$L POP=$POPN ARMS="hand dishann" MODE=extract "$@" sbatch --parsable --dependency=afterok:$T $P --export=ALL fillin-train.slurm)
  EXTRACT[$L]="${EXTRACT[$L]}:$E"; echo "$L $POPN: train=$T extract=$E"
}
train unident_s_cx partial SCRIPT_PARTNERS="potter server idle onion_potter tomato_potter"
train pantry_cx scripted
train unident_s_cx usual INCLUDE=unident_s_cx/fcp/s2/train-bench_sp-cx-lr.yml SCRIPT_PARTNERS=none
train pantry_cx usual INCLUDE=pantry_cx/fcp/s2/train-bench_sp-cx-lr.yml SCRIPT_PARTNERS=none
unset STEPS EXP_TAG

# Best responses at 5e6, split so each job runs 3 at a time for 2 rounds.
BR() { local J=$(LAYOUTS=$1 PARTNERS="$2" STEPS=5e6 EXP=br-5m CONCURRENCY=3 BR_LOGDIR=$HOME/Coding/masters-research/experiments/logs/br5m \
       sbatch --parsable --array=0 $P --export=ALL br-partners.slurm); echo "BR $1 [$2]: $J"; }
BR unident_s_cx "sp7_mid sp7_final sp9_mid sp9_final sp10_mid sp10_final"
BR unident_s_cx "sp12_mid sp12_final sp15_mid sp15_final sp16_mid sp16_final"
BR unident_s_cx "sp17_mid sp17_final sp19_mid sp19_final sp20_mid sp20_final"
BR pantry_cx "sp8_mid sp8_final sp9_mid sp9_final sp13_mid sp13_final"
BR pantry_cx "sp15_mid sp15_final sp17_mid sp17_final"

# Scoring, once each kitchen's arms are extracted.
declare -A ARMS5=([unident_s_cx]="partial-hand-5m partial-dishann-5m usual-hand-5m usual-dishann-5m"
                  [pantry_cx]="scripted-hand-5m scripted-dishann-5m usual-hand-5m usual-dishann-5m")
declare -A HO=([unident_s_cx]="7 9 10 12 15 16 17 19 20" [pantry_cx]="8 9 13 15 17")
declare -A EXPS=([unident_s_cx]="fcp-S2-partial-hand-5m fcp-S2-partial-dishann-5m fcp-S2-usual-hand-5m fcp-S2-usual-dishann-5m"
                 [pantry_cx]="fcp-S2-scripted-hand-5m fcp-S2-scripted-dishann-5m fcp-S2-usual-hand-5m fcp-S2-usual-dishann-5m")
for L in unident_s_cx pantry_cx; do
  D="--dependency=afterok${EXTRACT[$L]}"
  X=$(LAYOUT=$L ARMS="${ARMS5[$L]}" REF_S2="" S1_ARM="bench_sp-cx-lr bench_morl_ann-cx-lr" S1_SEEDS="1 2 3 4 5 6" HELDOUT_ONLY=1 SKIP_EXTRACT=1 \
      TAG=_sp_phase4_5m PARTNER_GROUP=heldout HELDOUT_ARGS="--heldout sp --heldout_exp bench_sp-cx-lr --heldout_seeds ${HO[$L]} --heldout_tags mid final" \
      sbatch --parsable $D $P --export=ALL fillin-crossplay.slurm)
  F=$(OUT=../experiments/results/fillin/$L-phase4-5m MANIFEST_ARGS="--s1_arms --s2_exps ${EXPS[$L]}" BASE=s2_usual-hand-5m \
      METRICS=fillin_metrics_${L}_phase4_5m.json REFERENCE_SEEDS="1 2 3" sbatch --parsable $D $P --export=ALL fillin-suite.sh $L)
  echo "$L scoring: crossplay=$X fillin=$F"
done
# unident_s_cx's mixed-dish held-out group (stage-1 MORL seeds, used by no population).
D="--dependency=afterok${EXTRACT[unident_s_cx]}"
M=$(LAYOUT=unident_s_cx ARMS="${ARMS5[unident_s_cx]}" REF_S2="" S1_ARM="bench_sp-cx-lr" S1_SEEDS="1 2 3 4 5 6" HELDOUT_ONLY=1 SKIP_EXTRACT=1 \
    TAG=_morlho_phase4_5m PARTNER_GROUP=heldout HELDOUT_ARGS="--heldout sp --heldout_exp bench_morl_ann-cx-lr --heldout_seeds 1 2 3 5 6 9 10 --heldout_tags mid final" \
    sbatch --parsable $D $P --export=ALL fillin-crossplay.slurm)
echo "unident_s_cx MORL-held-out crossplay: $M"
squeue -u $USER -h | wc -l; squeue -u $USER -h -r | wc -l
