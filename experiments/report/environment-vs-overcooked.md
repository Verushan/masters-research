# Our Overcooked environment vs the real game

Source: code reading of `zsc-eval` (fork of ZSC-Eval / Overcooked-AI), 30 Sep 2026. File:line citations are relative to
`zsc-eval/`. Claims about the Overcooked video games are general knowledge, labelled **[game]**, and approximate.

- **OLD env** = `zsceval/envs/overcooked/` (core MDP `overcooked_ai_py/mdp/overcooked_mdp.py`, wrapper `Overcooked_Env.py`).
  Used by `random0`, `random0_medium`, `random1`, `random3`, `small_corridor`, `unident_s` (`zsceval/overcooked_config.py:5-12`).
  **Every result in the reports is on this env.**
- **NEW env** = `zsceval/envs/overcooked_new/` (multi-recipe `*_m` / `*_mx` layouts).

## At a glance

| Feature | OLD env (unident_s etc.) | NEW env (`*_m`, `*_mx`) | Real Overcooked 1/2 **[game]** |
| --- | --- | --- | --- |
| Recipes | Onion soup only, exactly 3 onions | 5 recipes: OOO, OOT, OTT (value 20, cook 20); OO, OT (value 10, cook 10) | Many multi-step dishes |
| Order queue | None (`order_list=None`) | Fixed set of acceptable recipes, never changes, not observed | Timed ticket queue, new orders arrive |
| Order expiry / timers | Does not exist | Does not exist | Tickets expire; OC2 deducts points |
| Tips / bonus orders | None | `bonus_orders` (x2) in code, empty in every layout | Tips for speed, combo multiplier |
| Delivery reward | +20, shared by the team | Recipe value (20/10; `_mx` 10/30/30/5/15), shared | Points per dish plus tip |
| Wrong-dish penalty | None: every soup is valid | -10 for a recipe not in the order set | Varies |
| Other penalties | None (idling, collisions, time all free) | None | Expired orders; burning / fire costs time |
| Shaped reward (training only) | Onion in pot +3, useful dish pickup +3, soup pickup +5, per agent, annealed to 0 | Same values; pot +3 only if the placement is useful | None |
| Cooking start | Automatic at 3 onions | INTERACT with empty hands; 1-3 ingredients | Automatic |
| Cook time | 20 steps | 20 or 10 steps | Real-time seconds |
| Burning / fire | Does not exist: soup waits forever | Does not exist | Exists |
| Chopping / frying / mixing | Does not exist | Does not exist | Exists |
| Plate washing | Does not exist: infinite dishes | Does not exist | Dirty plates return, sinks |
| Throwing / dashing | Does not exist | Does not exist | Exists (OC2) |
| Dynamic kitchens | Static grid | Static grid | Moving platforms, ice, portals |
| Movement | Discrete grid, 6 actions; a collision sends both agents back | Same | Continuous real-time, 1-4 players |
| Episode | 400 steps, no early termination, fixed start state | Same | Level timer (minutes), star thresholds |
| Time / orders in observation | Neither (time only with `--use_timestep_feature`) | Binary "urgency" plane in the last 40 steps; no orders | Both on the HUD |
| Observation (`ppo` encoding) | 20 channels, x255 | 25 channels | Pixels |

**In one line:** the environment is a stripped-down Overcooked. There is one recipe, orders never change or expire, and
there are no penalties, fires, plates to wash, or chopping. The only pressure is the 400-step clock, which the agent cannot
see. Coordination difficulty comes entirely from the layout and from sharing pots, dispensers and corridors.

## Details

### Recipes and orders

**OLD env**
- Soup types are `["onion", "tomato"]` (`overcooked_mdp.py:40`). The layouts set `num_items_for_soup: 3`,
  `cook_time: 20`, `delivery_reward: 20` and `start_order_list: None`. The wrapper forces `start_order_list: None`
  (`Overcooked_Env.py:574`).
- An order-list mechanism exists but is unused. With it, a delivery would pay only if it matched the front order, and an
  empty list would end the episode (`:904-914`, `:628-632`). With `None`, every soup pays `delivery_reward` (`:904-905`).
- Orders are never in the observation: "currently not including time left or order_list in featurization" (`:1261`).
- There is no timer, expiry, bonus or tip anywhere in the mdp or env code (grepped for
  `expir|burn|fire|wash|chop|throw|tip|deadline|time_left`).
- Tomato is unusable in the old env: no old layout has a `T`, and the code would raise if one did (`:834`, `:1239-1240`).

**NEW env**
- Recipes have at most 3 ingredients, onion and tomato (`overcooked_mdp.py:57-62`).
- Each `_m` layout lists `start_all_orders` OOO, OOT, OTT, OO, OT, with `recipe_values` [20, 20, 20, 10, 10],
  `recipe_times` [20, 20, 20, 10, 10] and no bonus orders.
- Orders are static; the code caches because "recipe values are currently static (i.e. bonus_orders doesn't change)"
  (`:1821-1823`). The bonus mechanism multiplies by `order_bonus=2`, but every layout's bonus list is empty.

### Rewards

- **Sparse:** +20 per delivery in the old env (`overcooked_mdp.py:876-881`, `:904-905`). It is summed over agents and
  shared, so each agent receives `sparse_team + shaping_factor * own_shaped` (`Overcooked_Env.py:166`, `:1365-1371`).
- **NEW env:** −10 for delivering a recipe not in `all_orders` (`:1463-1465`, marked "TODO: penalty"). Bonus orders
  would be paid double. These are the only penalties in either env.
- **No penalty** for idling, collisions or time. STAY and idle moves are only counted in `shaped_info`, and pay only
  under HSP bias weights.

**Hand-shaped reward** (`BASE_REW_SHAPING_PARAMS`, `overcooked_mdp.py:373-380`):

| Event | Value | Paid when |
| --- | --- | --- |
| Placement in pot | +3 | An onion goes into a non-full pot (`:859-875`) |
| Dish pickup | +3 | Only if cooking or ready pots outnumber held dishes and no dish is on a counter (anti-hacking, `:835-843`) |
| Soup pickup | +5 | A dish is used on a ready pot (`:846-854`) |

The shaping factor is annealed linearly from 1 to 0 over `reward_shaping_horizon` steps (`Overcooked_Env.py:713-721`).
All our 2e6-step runs set that horizon to the run length.

### Cooking (OLD env)

- A pot starts cooking automatically when its third onion lands (`:958-972`); no interact is needed.
- Cook time is 20 steps, capped. A ready soup waits forever: there is no burning or overcooking (`:965`).
- A pot with fewer than 3 onions never cooks and cannot be emptied. A 4th onion is refused.
- In the NEW env, cooking starts with an INTERACT and can use 1-3 ingredients; nothing burns there either.

### Actions and movement (both envs)

- Six actions: N, S, E, W, STAY, INTERACT (`actions.py:39-44`).
- Moving into a blocked tile turns the agent in place.
- If both agents end on the same cell, or try to swap cells, **both** go back to where they were (`:943-956`).
- Interacts resolve first (player 0, then player 1), then movement, then the pots tick (`:762-776`).
- Interact acts on the faced tile:
  - Counters hold one item.
  - Onion and dish dispensers are infinite and need empty hands.
  - The serving tile accepts only soup.
  - The plate disappears on delivery (`:902`).
- None of these exist: dish washing, chopping, frying, fire, throwing, dashing.

### Episode and observation

- **Episode:** 400 steps in every training script (the argparse default is 200). There is no early termination.
- **Start state:** fixed in the old env (`:540-543`). Seats swap only with `--random_index`, which is off by default.
- **Old `ppo` encoding:** 20 channels (`lossless_state_encoding`, `:1141-1270`), multiplied by 255
  (`Overcooked_Env.py:698-701`):
  - self and partner position;
  - 4 + 4 orientation planes;
  - pot, counter, onion dispenser, dish dispenser, serving tile;
  - onions in pot, cook time, soup, dishes, onions.
- **No time left and no orders** in the observation.
- The centralised critic sees both agents' views (40 channels).

### Layouts

- **unident_s (Asymmetric Advantages)**, where all headline results are measured:

  ```
  XXXXXXXXX
  O XSXOX S
  X   P 1 X
  X2  P   X
  XXXDXDXXX
  ```

  A pot column splits the kitchen into two halves. Each half has its own onion dispenser, dish dispenser and serving
  tile, and both halves reach both pots. No counter hand-off is possible, so each cook can run the whole loop alone, and
  the only interaction is through the two shared pots: who fills or plates which pot, without over-filling or racing
  for the same soup.

- **random0 (Forced Coordination):** a counter column separates the agents. One side has the onions and dishes, the
  other has the pots and the serving tile, so everything must be handed across.
- **random1 (Coordination Ring):** a ring of one-cell corridors around a central counter. The agents must agree on a
  direction of travel or they block each other.
- **random3 (Counter Circuit):** a long loop around a central island, with pots at the top, onions at the bottom and
  dishes and serving at the sides. Blocking in the loop is the main difficulty. Passing onions over the island is a
  strong but non-obvious strategy.

### What this fork adds (mechanics untouched)

Neither core MDP file is modified. The fork adds:

- the MORL objective / task-vector reward hook (`zsceval/envs/morl/`);
- optional observation channels;
- scripted partners, including a new `idle` script, and **mid-episode partner swaps**
  (`--script_swap_steps/_pool/_prob`, `Overcooked_Env.py:1257-1267`);
- the `_mx` mixed-incentive layouts.

## Where the real game differs **[game]**

- **Orders:** a visible queue of tickets, each with a countdown, and new tickets during the level. Expired tickets cost
  points in OC2. Our env has no orders at all.
- **Tips and combos** for fast, in-sequence service: absent.
- **Recipes** are multi-step (chop, then cook, fry, mix or bake, then assemble): ours is 3 raw onions into a pot.
- **Plates** come back dirty and need washing: ours are infinite.
- **Fire:** food burns and pots catch fire: our soup waits forever.
- **Kitchens and movement:** moving or splitting kitchens, ice, portals; continuous movement, dashing, throwing; up to
  4 players. Ours is a static grid of up to 13x5, with 2 players and 6 discrete actions.
- **Scoring:** a real-time level timer with star thresholds and time and orders on the HUD. Ours is 400 discrete steps,
  a sum of +20 deliveries, and no time or order information in the observation.

**Implication for the thesis:** partner-adaptive behaviour in our env can only show up as the division of the four
kitchen jobs (fill, fetch dish, plate, deliver) and use of space. There is no order-priority or deadline dimension to
adapt on, so a partner who neglects work is the main source of adaptation pressure. That is why the fill-in test
constructs neglect directly.
