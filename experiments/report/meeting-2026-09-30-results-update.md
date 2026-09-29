# Supervisor meeting 30 Sep — results that landed 29 Sep

Paste-ready for the meeting doc (15aaa70a), which is not shared with Claude.
Every job the handover listed (61568–61587) finished with exit 0 and non-empty output.
Sources: `experiments/results/metrics_unident_s_hsp_scripted2.json`,
`metrics_unident_s_hsp_scripted_ns.json`, `metrics_random3_hsp_scripted.json`,
`fillin_metrics_unident_s_step8.json`, `fillin_metrics_random3_step8.json`.
Zero-shot means come from `experiments/zsc_by_arm.py`. Every p-value is an exact permutation test at the seed level.

## Running now: status

| Status | Job | What it is | Outcome |
| --- | --- | --- | --- |
| Done | 61568 | neglect2 runs 6–10 | neglect2 is at 10 runs |
| Done | 61569 | Zero-shot check of the corrected arms | Below: ties A |
| Done | 61587 → 61583 → 61584, 61586 | Q3 ablation: arm A without swaps (`handns`), 10 runs | Below: the specialists do the work |
| Done | 61572, 61574 → 61580 → 61581, 61582 | Q4: arm A and noann2 on random3, 10 runs each, plus the FCP reference | Below: invalid, every trained agent floors |

## Update 29 Sep: the corrected MORL arms tie A zero-shot but still lose at fill-in. Verdict: not worth it (final)

Zero-shot mean return against the 16 held-out HSP partners on unident_s, 10 runs per arm:

| Agent | Mean | SD | Worst | Best | vs A |
| --- | --- | --- | --- | --- | --- |
| noann2 (corrected rule, bonuses never fade) | 238.5 | 45.1 | 140.6 | 281.2 | +10.1 (p = 0.55) |
| neglect2 (corrected rule, bonuses fade) | 232.8 | 45.0 | 126.2 | 273.8 | +4.4 (p = 0.80) |
| A (hand-shaped, scripted specialists) | 228.4 | 24.6 | 169.4 | 251.9 | — |
| Usual stage-2 agent | 142.9 | 37.9 | 65.0 | 188.1 | −85.4 (p < 0.001) |

The corrected arms pass the zero-shot criterion ("does not score lower than A"). They are nominally ahead, but their run-to-run spread is almost twice A's, and their worst runs are 30–43 points below A's worst.

neglect2 now has all 10 runs, and it fails the fill-in criteria more clearly than noann2 does. Against A:

- soups against neglecting partners: −0.20 per 100 steps (p = 0.029)
- soups against held-out partners: −0.30 (p = 0.002)
- soups after held-out swaps: −0.36 (p < 0.001)
- neglected work covered after held-out swaps: −9 points (p = 0.002)
- 29 steps to switch after a swap, against A's 21

noann2 is unchanged from 28 Sep: it ties A in single-partner games, and trails after held-out swaps by 0.24 soups (p = 0.006) and 7 points of coverage (p = 0.02).

**Q2 is answered.** Both corrected arms fail "C beats A on soups after swaps and on coverage", and both are significantly worse there. By the criteria fixed in advance, and for the corrected rule as well, neglect-weighted MORL is not worth it. The zero-shot tie shows the MORL reward does not hurt transfer to learned partners. It does not show that it helps.

## Update 29 Sep: Q3, specialists or swaps? The specialists

Arm `handns` is arm A trained against the same five specialists with no mid-game swaps (10 runs). Against A:

| Measure | A | handns | Difference |
| --- | --- | --- | --- |
| Zero-shot vs 16 HSP partners | 228.4 | 217.9 | −10.4 (p = 0.48) |
| Soups vs neglecting partners | 2.84 | 2.83 | −0.02 (p = 0.91) |
| Soups vs held-out partners | 2.93 | 3.02 | +0.09 (p = 0.19) |
| Soups after held-out swaps | 2.42 | 2.25 | −0.17 (p = 0.19) |
| Neglected work covered after held-out swaps | 67% | 60% | −7 pts (p = 0.041) |
| Steps to switch after a swap | 21 | 26 | — |
| Role set by starting seat | 0.61 | 0.41 | −0.20 (p = 0.057) |

Almost all of arm A's gain comes from who it trains with. Without swaps it keeps the zero-shot score (217.9, against 142.9 for the usual stage-2 agent) and the single-partner fill-in. The one significant cost is 7 points of coverage after an unfamiliar swap; the switch also takes 5 steps longer. The seat habit is weaker without swaps (0.41 against 0.61, p = 0.057). A plausible reading is that swaps hand the new partner the seat's usual role, so the agent learns the seat as well as the partner.

## Update 29 Sep: Q4, second kitchen. random3 did not train, so the question is still open

On random3, every trained agent scores at the floor. This is a training failure and says nothing either way about the methods:

- Zero-shot against the random3 HSP partners: A 0.1, noann2 0.0, usual stage-2 11.6, stage-1 3.3.
- The random3 HSP partners score 0 with each other and 0 with every trained agent. With the partner they were trained beside, they reached 66–190, so each one learned a private protocol that works only with that partner.
- Fill-in: the oracle makes 1.34 soups per 100 steps against neglecting partners. A makes 0.21, noann2 0.13, the usual stage-2 agent 0.34. Beside an idle partner, A and noann2 make 0 soups from either seat, so they cannot run the kitchen alone.
- Every stage-2 agent on random3 scores 0 in self-play, including the usual FCP reference. Stage-1 self-play agents reach 107 (460 on unident_s).
- The agents' moves never became predictable: action entropy is 0.6–1.1, against 0.3–0.5 on unident_s. Training eval flattened at 40–50 by 1M steps.

random3 is Counter Circuit, the hardest of the old layouts to coordinate on. Our 2M-step budget, sized on unident_s, is a fifth of upstream's 1e7. Before re-running the arms there, a two-seed pilot of arm A at 1e7 steps would show whether budget is the cause. It needs about 9 hours on the cluster. The zero-shot measure on random3 also needs HSP partners that work with more than one partner, and this pool's are not.

## Questions for you, updated

1. **Finding 1 in the thesis.** Q3 now says the specialists, not the swaps, account for the gain. Our suggestion stands: report it as a finding in its own right, and make it the bar for every MORL arm.
2. **Ending the MORL line.** The corrected arms have now lost to A on the fill-in criteria. Our suggestion stands: accept the negative result. The zero-shot tie goes in as "does no harm".
3. **Second kitchen.** random3 as run cannot answer the question. Options: (a) random3 at 1e7 steps, with a two-seed pilot first (~9 h), then 10+10 runs (~2 days); (b) a layout that trains at 2e6. unident_s is the only one we know does; random0 floors for everyone for a different reason (forced coordination).
4. **Reliability lead (Finding 4).** It is still one kitchen, and the second-kitchen check is blocked by question 3.
