# Pre-registration: trained without swaps, tested with them (15 Oct 2026)

Written before any result of the runs below exists.

## Question

The supervisors asked (15 Oct): agent A learned well from specialists that were swapped mid-game during
training. Does MORL show promise in *uncertain* situations, that is, when an agent trained **without** swaps
meets a partner who changes mid-game?

## Design

A 2 x 2 of ego reward x swap-training. Every agent trains against the same five scripted specialists (potter,
server, idle, dial2, dial8), stage 2, 2e6 steps, 12 rollout threads, 10 runs per cell.

| | Trained with swaps (step 200, p = 0.5) | Trained without swaps |
| --- | --- | --- |
| Hand-shaped reward | A (`scripted-hand`, existing) | A-ns (`scripted-handns`; existing on unident_s, new on random1) |
| Corrected neglect-weighted MORL | C2' (`scripted-noann2`, existing) | C2'-ns (`scripted-noann2ns`, new on both kitchens) |

Kitchens: **unident_s** (primary) and **random1** (replication).

## Tests (all with swaps)

1. **Zero-shot with a partner swap** (primary). `eval/cross_play.py --swap_step 200`, sampled actions, 5 games per
   pair and seat, both seats. At step 200 the held-out partner is replaced by the next held-out partner; the agent
   is not told. Held-out partners: the 16 HSP agents on unident_s; on random1 the self-play held-out set
   (`bench_sp` seeds 7-12, `mid` and `final`), because random1's HSP partners score 0 with anyone but their own
   training partner. Outcome per run: sparse return from step 200 on (`after`), averaged over partners, seats
   and games (`experiments/analyze_swaps.py`).
2. **Fill-in with swaps** (secondary). The fill-in suite's swap schedules, including the two never seen in
   training (`server>idle>potter`, `dial5>clutter>generalist`). Outcomes: soups per 100 steps after a held-out
   swap (`heldout_soups_after_swap`), coverage after the swap, and takeover latency (`analyze_fillin.py`).

The training run is the unit; tests are exact or 20,000-sample permutation tests, two-sided, alpha = 0.05.

## Hypotheses

- **H1 (the supervisors' question).** C2'-ns > A-ns on the primary outcome: among agents trained without swaps,
  MORL holds up better when the partner changes.
- **H2.** The benefit of swap training is smaller for MORL: (A - A-ns) - (C2' - C2'-ns) > 0 on the primary
  outcome, i.e. MORL needs swap training less.

## Decision rule

We will say MORL **shows promise in uncertain situations** only if H1 holds on unident_s (p < 0.05, C2'-ns
ahead) **and** the random1 difference has the same sign. Otherwise the result is reported as no evidence that
MORL helps with partner changes, alongside the effect sizes and intervals. H2 is reported either way.

## Known limits

- C2' and C2'-ns see their MORL weights in the observation (`--use_morl_obs_weights`); A and A-ns do not. This
  is how the neglect rule works and is unchanged from the 7 Oct comparisons.
- A-ns on unident_s was trained on 29 Sep; the new runs use the same code path for this arm (verified by the
  unchanged `fillin-train.slurm` reward lines) but a later checkout.
