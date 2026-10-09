# Design spec: an Overcooked environment with timed orders

Status (9 Oct 2026): layers 1-6 implemented and validated (see CLAUDE.md, "Timed-order environment"). Defaults were
retuned to arrival 30 / deadline 75 on scripted pairs (experiments/results/timed_orders_check.json); the table below
keeps the first guesses. Layer 7 (HSP order bias terms) is not done. Originally a proposal for the 7 Oct meeting.

## Why

In the environment every result so far uses, all objectives serve one goal: more soup. There are no orders, no
deadlines and no penalties, and the clock is invisible (`experiments/report/environment-vs-overcooked.md`). That
leaves two gaps:

1. **MORL has nothing to trade off.** A reward vector whose entries all point the same way adds no information over
   the scalar. That may be the deepest reason MORL never helped. Timed orders create real conflicts: rush the
   expiring order or batch two, cook fast or cook the valuable recipe, help the partner's order or start a new one.
2. **Partner adaptation has little to work with.** Today a lazy partner only means less soup. With orders, it also
   means missed deadlines, and what the agent should do depends on *which* order is at risk.

So this is the fairest test of MORL we can build, and a harder test of the specialist route.

## What exists to build on

The multi-recipe environment (`zsceval/envs/overcooked_new/`, the `*_m` layouts) already has:

- `Recipe` with onion/tomato combinations, per-recipe values and cook times (`overcooked_mdp.py:57`);
- `OvercookedState` with `all_orders` and `bonus_orders` (`:676-702`). They are static: `get_recipe_value`
  (`:1448`) caches on the assumption that they never change, and bonus orders are empty in every layout;
- `deliver_soup` (`:1493`) and `get_recipe_value`, including a −10 for unlisted recipes;
- a lossless observation (25 channels) with an "urgency" plane in the last 40 steps;
- the objective-vector hooks the MORL arms use.

It does **not** have: an order queue, arrival over time, deadlines, expiry, lateness penalties or speed-scaled
pay. Its scripted agents are also broken: `script_agent/utils.py:33` reads `mdp.num_items_for_soup`, which the
multi-recipe MDP does not define. Throwing does not exist in either environment.

## Mechanics to add

| Mechanic | Proposed rule | Parameter (first guess) |
| --- | --- | --- |
| Order queue | Up to `Q` open orders, each `(recipe, time_left, value)` | `Q = 3` |
| Arrival | A new order every `A` steps (random within ±25%) while the queue has room; recipe drawn from the layout's recipe set | `A = 60` |
| Deadline | Each order expires `T` steps after arrival | `T = 150` |
| Expiry penalty | An expired order is removed and costs the team `P` | `P = 10` |
| Delivery match | A delivered soup fills the open order with the same recipe that is closest to expiring; no match pays 0 (unlisted recipes keep −10) | — |
| Speed-scaled pay | `value × (0.5 + 0.5 × time_left / T)`: a soup delivered at arrival pays full value, one delivered at the deadline pays half | — |
| Episode | 400 steps as now; the remaining-time plane is always on | — |
| Throwing | **Deferred.** It changes movement and collision rules and adds speed, not a new trade-off | — |

These parameters will need tuning. A scripted oracle that reads the queue should keep the queue mostly served,
and an idle partner should cause frequent expiries. That spread is what gives partner adaptation room.

## Changes, by layer

1. **State** (`OvercookedState`): add `orders: list[(recipe, arrival, deadline)]` and an RNG seed for arrivals.
   Update `to_dict`, `from_dict`, equality and hashing so traces and replays still work.
2. **Dynamics** (`step_environment_effects`): tick order timers, expire and penalise, spawn arrivals.
   `get_recipe_value` and `deliver_soup` match against the queue, and the static-order cache is removed.
3. **Reward** (wrapper `Overcooked_Env.py`): sparse reward = speed-scaled order pay − expiry penalties.
   New `shaped_info` counters: `order_delivered`, `order_expired`, `on_time_ratio`.
4. **Observation**: for each queue slot, constant planes for onion count, tomato count and time-left fraction
   (`Q × 3 = 9` planes), plus a remaining-episode plane. The observation width changes, so **every agent must be
   retrained in this environment**; no existing checkpoint loads.
5. **MORL objectives** (`envs/morl/objectives.py`): add `on_time_delivery`, `speed_bonus`, `expiry_avoidance`
   (≤ 0) and `recipe_value`. These genuinely pull against each other, which is the point.
6. **Scripted partners**: fix the multi-recipe scripts (the `num_items_for_soup` bug) and add order-aware
   routines ("cook the most urgent order"). The specialist family gains an order-preference knob: urgent first,
   valuable first, or ignore orders.
7. **HSP held-out partners**: extend the bias space with order-related events (expiry sensitivity, speed
   seeking), so the held-out set covers the new behaviours.

## Validation before any training

- **Unit checks:** order lifecycle (arrival, matching, expiry, penalty), speed-scaled pay at both ends of the
  window, replay determinism with a fixed seed.
- **Scripted oracle:** a queue-reading oracle must keep the expiry rate low on every layout we use. An idle
  partner must make it high.
- **Trap from this project:** score with sampled actions as well as greedy ones from the start. Greedy play
  deadlocked on random1's narrow ring.

## Cost

| Step | Estimate |
| --- | --- |
| Mechanics, observation and reward (layers 1–4) | 3–4 days |
| MORL objectives, fixed and order-aware scripts (5–6) | 1–2 days |
| Validation | 1 day |
| Retraining in the new env: HSP partners, stage-1 population, stage-2 arms, best responses | 3–4 days of cluster time |
| Scoring (sampled actions, BR-Prox, fill-in) | 1 day |
| **Total** | **about 2 weeks** |

## Decisions needed from the supervisors

1. Is the order mechanic (queue, deadlines, penalties, speed pay) the right first extension, with throwing deferred?
2. One layout or several? Suggest unident_s_m and random1_m: the two kitchens where our results already hold.
3. Should the specialist family come first (about 1 week), so the new environment gets the stronger baseline from
   day one?
