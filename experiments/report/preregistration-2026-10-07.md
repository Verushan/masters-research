# Pre-registered criteria for the 7 Oct 2026 report

Written 30 Sep 2026, after the jobs below were submitted and before any of their results existed. Unless stated
otherwise, every test is an exact (or, when too many splits exist, 20,000-sample) permutation test with the training
run as the unit. alpha = 0.05, two-sided.

## 1. Partner removal (jobs 62019 on unident_s; K = 5 is the existing arm A / C2′ runs)

**Design.** Arms A (hand-shaped) and C2′ (corrected neglect-weighted MORL, bonuses stay) are each trained against
K = 1, 2, 3, 4 of the five specialists, with 5 seeds per (arm, K). Seed s uses specialists s, s+1, …, s+K−1 (mod 5), so
each specialist appears K times per K and both arms see the same subset for the same seed. With K = 1 there are no
swaps. K = 5 is the existing 10 runs per arm. Budget, schedule and evaluation are unchanged.

**Measures.**
- Primary: zero-shot mean return against the 16 held-out HSP partners (deterministic pass).
- Secondary: fill-in soups per 100 steps after held-out swaps.

**Questions and tests.**
1. *Do partners matter for A?* The slope of A's zero-shot score over K = 1…5, tested by permuting K labels across A's
   runs. Positive and significant = the specialist result depends on partner diversity.
2. *Does MORL gain more from more partners?* The supervisors' hypothesis. The test statistic is
   slope(C2′) − slope(A) over K. Significance comes from permuting the arm label within each (K, seed) pair, which
   keeps the paired design.
   - Positive and significant = MORL makes better use of more partners.
   - Negative and significant = A does.
   - Otherwise: no difference detected.
3. *Unfair comparison, as asked.* For each K < 5, A(K) is compared with C2′(K = 5). We report the smallest K at which
   A(K) is significantly below C2′(K = 5). This is descriptive: pool size and reward change together here, so it
   cannot be read as an effect of the reward.

## 2. MORL reward on the stage-2 trainee (unident_s top-up seeds 13–20, random0 seeds 7–12, random1 seeds 1–12)

The arms are the hand-shaped trainee (`fcp-S2-bench_sp`) and the annealed team-credit MORL trainee
(`fcp-S2-bench_sp-annego`). Each kitchen trains both against the same 18-partner stage-1 population. The measure is
zero-shot mean return against that kitchen's 16 HSP partners.

- **"Better stage-2 agent" (mean).** MORL − hand, pooled over kitchens. Each kitchen's difference is scaled by that
  kitchen's pooled SD. The test permutes arm labels within kitchen. The claim needs p < 0.05 pooled.
- **"More reliable" (spread).** log(SD_MORL / SD_hand), averaged over kitchens, with the same stratified permutation.
  The claim needs p < 0.05 pooled **and** the same direction in at least 2 of the 3 kitchens.
- **Replication of the existing lead.** The unident_s spread result (SD 15.2 vs 34.8 at 12 runs) was found after the
  fact. The new unident_s seeds 13–20 are an independent sample, and the lead replicates if SD_MORL < SD_hand among
  them alone. At n = 8 per arm this is reported with its p-value but is underpowered by design.
- Each kitchen's result is also reported alone. A kitchen where either arm's mean is below 20 is flagged as floored
  (random0 was near the floor before), and its spread is not interpreted.

## 3. Specialist training on a second kitchen (job 62020, random1, arms A and C2′, 10 runs each)

This is the random3 question again, on a kitchen where the scripted partners do cook (checked 30 Sep: the oracle
makes 5–10 soups per game beside every partner except an idle one from seat 0).

- The Finding 1 claim replicates if A's zero-shot score exceeds the usual stage-2 agent's (seeds 1–12 above), with
  p < 0.05.
- The Finding 2 verdict is re-checked with the same fill-in criteria as unident_s: C2′ must beat A on soups after
  held-out swaps **and** on coverage.
- If A or the usual stage-2 agent floors (mean zero-shot below 20), random1 is reported as uninformative, as random3 was.

## 4. Behaviour similarity

This part is exploratory and was run on existing agents before this file was written
(`experiments/behaviour_similarity.py`, `experiments/action_agreement.py`). Its p-values are reported as descriptive.
