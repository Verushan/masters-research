# Pre-registration: measuring convergence, decline and drift (10 Nov 2026)

Written before any result of the hyperparameter sweep (`*-cx-lr-{ppo5,mb4,lrdecay,clip01,shapekeep,ent0}`)
or the MORL ablation (`bench_morl_ann-cx-lr-{norq,nocoord,noprep}`) has been looked at. Every
"learned / held / collapsed" count reported before this date was read by eye off smoothed W&B training curves;
from here on they come from the rules below, applied by `experiments/convergence_audit.py`.

## Why the training curve is not enough

- The logged `ep_sparse_r` averages the ~12 training episodes that ended since the last log, so one point is noisy.
- Choosing the peak on that same noisy curve inflates it (winner's curse), which inflates every peak-to-final drop.
- A falling return does not say whether the policy changed suddenly or drifted, or whether it fell on the training
  partners or only on held-out ones.

## Measurement

For every run, about 25 evenly spaced saved checkpoints (stage 1: every 3rd of ~74; stage 2: about every 2nd of
~55) are replayed offline with sampled actions, as every zero-shot number in this project.

- **Partners.** Stage 1: the checkpoint plays itself. Stage 2: each of its training partners (scripted specialists
  or the frozen population), once in each seat.
- **Two independent game sets, A and B**, each 8 self-play games (stage 1) or one game per partner and seat
  (stage 2). Seeds are fixed per set and identical for every checkpoint of every run (common random numbers), so
  checkpoint-to-checkpoint differences are not start-state or sampling noise.
- **Noise floor.** For every checkpoint, |mean A − mean B| is recorded. Its 95th percentile is reported per
  kitchen; no decline smaller than it is interpreted.

## Classification of a run (fixed thresholds)

Let `R_A(c)`, `R_B(c)` be the mean sparse return of checkpoint `c` on set A and B, and `C = 50` the competence bar
already used for held-out partners.

1. **Peak** `c*` = the checkpoint maximising the 3-checkpoint running mean of `R_A`. Its value is confirmed on
   set B: `P = R_B(c*)`. The peak is never both chosen and measured on the same games.
2. **Final** `F` = mean of `R_B` over the last 3 checkpoints.
3. **Not converged**: `P < C`.
4. **Declined**: converged, `F <= 0.75 P`, and a one-sided Welch test on set-B games (peak vs last 3 checkpoints)
   gives p < .05.
5. **Collapsed**: declined and `F < C`.
6. **Stable**: converged and not declined.

**Collapse time** = the first checkpoint after `c*` from which the running mean of `R_B` stays below `0.75 P` to
the end. Runs that never collapse are censored at their last checkpoint.

## Drift (behaviour, not return)

A state bank is recorded from 4 games of the peak checkpoint (same partners as above). For every checkpoint `c`
the mean KL(π_peak ‖ π_c) over the bank is computed (recurrent agents are teacher-forced through the bank's
observation sequences with their own memory), as is the KL between consecutive checkpoints.

- **Sudden** change: one consecutive-checkpoint KL greater than half the total KL accumulated from peak to final.
- **Gradual** drift: otherwise.

## Arm-level comparisons

- Rates of converged, declined and collapsed runs per arm, with exact (Clopper–Pearson) 95% intervals; arms
  compared with Fisher's exact test against the base (`bench_sp-cx-lr` for the sweep, `bench_morl_ann-cx-lr` for
  the ablation).
- Time to collapse: Kaplan–Meier curves, log-rank test against the base.
- Peak and final return: exact permutation test on run-level values (`compare_arms.permutation_test`).
- Stage 2 only, secondary: the same curve beside held-out HSP partners, to check that the checkpoint chosen on the
  training partners (`--checkpoint best`) is also near the best on partners it never met.

## Validation of the cheap measure

The same rules are applied to the W&B training curve (`ep_sparse_r`, 3-point running mean, peak and final from
the one curve). Agreement with the offline classification is reported (Cohen's kappa). Only if kappa >= 0.8
may later arms be classified from training curves alone.

## Decision rules for the sweep

A variant **reduces collapse** if its collapse rate is lower than the base's with Fisher p < .05, or its
Kaplan–Meier curve differs with log-rank p < .05 in the right direction. With 12 runs per arm, only large effects
can pass. A variant that does not pass is reported as "no detectable effect", not as "no effect".

Interpretation fixed in advance:

- `ppo5`, `mb4`, `clip01` or `lrdecay` pass and `shapekeep` does not: collapse is a step-size problem.
- `shapekeep` passes and the step-size variants do not: it is a recovery-signal problem.
- `ent0` passes: the entropy bonus drives the drift back to random after the break.
- Collapses classified "sudden" support a single destructive update; "gradual" ones support drift.
