"""Annotate every W&B run with what it was, and list the ones worth deleting.

Each run gets
  notes  a plain description of the experiment (stage, reward, partners,
         budget, where its results are used), then `host: <machine>` -- the
         machine is what the notes field held before;
  tags   stage-{1,2,hsp,br,legacy}, study-<name>, kitchen-<layout> and one
         status-<...>:
           used          checkpoints in the cluster policy pool, or a BR / HSP
                         partner the scoring reads
           pilot         smoke tests, verifications and pilots
           legacy        the upstream pipeline shakeout (May-Aug)
           superseded    replaced by a later version (already tagged so, or a
                         rule below)
           crashed       crashed / failed: never extracted, nothing reads them
           unreferenced  finished, but nothing in the pool or results uses them
           in-progress   finished, but sibling seeds are still training (re-run later)
Existing tags are kept. `unused` and `hidden` matter -- extract_S2_models.py
drops runs carrying them -- so this never adds or removes either.

Dry run by default: prints the plan. --apply writes notes and tags.
--delete CATEGORY... deletes runs whose status is in the given categories
(e.g. `crashed pilot`), after printing them; it asks for confirmation unless
--yes. Runs that are still running are never touched.

    python util/wandb_catalog.py --inventory runs.jsonl --pool pool.txt
    python util/wandb_catalog.py --inventory runs.jsonl --pool pool.txt --apply

`--inventory` is a JSON-lines dump of the runs (written by --dump); `--pool` is
`find policy_pool -maxdepth 4 -type d` from the cluster, relative to the pool.
"""

import argparse
import collections
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# What each experiment name means. First match wins; {layout} is filled in.
# ---------------------------------------------------------------------------

MORL_SET = {
    "": "default objectives (task, prep, plating, coordination)",
    "anc": "anchored objectives (coordination credited only once the handed item is used)",
    "ancval": "anchored objectives, task priced at the delivery reward (20)",
    "live3": "anchored_live3 objectives (task, prep, plating; no coordination)",
    "rec": "recipe objectives (multi-recipe env)",
    "obs": "anchored objectives, live w appended to the observation",
    "anct": "anchored objectives, dense terms annealed, team task credit",
    "live3t": "anchored_live3 objectives, dense terms annealed, team task credit",
    "smoke": "smoke test",
}
S1_ARMS = {
    "bench_sp": "hand-shaped reward (sparse + annealed shaping) -- the ZSC-Eval baseline arm",
    "bench_sparse": "sparse reward only (MORL weights 20,0,0,0)",
    "bench_morl": "MORL reward w.r_vec, uniform fixed w",
    "bench_morl_ad": "MORL reward, mirror-descent adaptive w",
    "bench_morl_ad_obs": "MORL reward, adaptive w shown to the agent",
    "bench_morl_ann": "MORL reward with the dense objectives annealed like the baseline's shaping",
    "bench_morl_fill": "MORL fill rule: annealed objectives, complement-adaptive w, partner objective shares in the obs",
    "bench_sp_shares": "hand-shaped reward plus the partner's objective shares in the obs (ablation)",
    "bench_morl_div": "MORL reward, a different fixed w per seed (population diversity)",
}
S2_REWARD = {
    "hand": "hand-shaped reward",
    "handns": "hand-shaped reward, no mid-game swaps (A-ns)",
    "neglect": "original neglect-weighted MORL reward, annealed (C)",
    "noann": "original neglect-weighted MORL reward, not annealed (C')",
    "neglect2": "corrected neglect-weighted MORL rule (prior 0.5, halflife 10), annealed (C2)",
    "noann2": "corrected neglect-weighted MORL rule, not annealed (C2')",
}
SPECIALISTS = "5 scripted specialists (potter, server, idle, dial2, dial8), swapped at step 200 with p=0.5"


def describe(layout, exp, tags):
    """(stage, study, description, default status) for one experiment name."""
    if exp in ("sp",):
        return "legacy", "upstream-fcp", "Upstream ZSC-Eval FCP stage-1 self-play, pipeline shakeout (1e6 steps).", "legacy"
    if exp == "fcp-S2-s16":
        return "legacy", "upstream-fcp", "Upstream ZSC-Eval FCP stage-2 over 16 'sp' agents, pipeline shakeout.", "legacy"
    if exp in ("morl_verify", "sp_verify"):
        return "legacy", "upstream-fcp", f"Verification run ({exp}) from the first MORL wiring, Aug 2026.", "pilot"
    if exp == "morl":
        return "1", "morl-early", "First MORL self-play test (before the MORL benchmark existed).", "legacy"
    if exp in ("hsp-s1", "hsp-s1-pilot"):
        pilot = exp.endswith("pilot")
        return "hsp", "hsp-partners", (
            "HSP bias-agent pair: w0 trained on a randomised event-weight reward, w1 a sparse-reward partner. "
            "w0 is one of the held-out partners the zero-shot score and BR-Prox are measured against."
            + (" Pilot at the 2e6 budget." if pilot else "")
        ), "pilot" if pilot else "used"
    if exp == "br":
        return "br", "br-prox", (
            "Best response: an agent trained against one held-out HSP partner only (2e6 steps). "
            "Its return is the denominator of ZSC-Eval's BR-Prox for that partner."
        ), "used"
    m = re.match(r"^(bench_[a-z_]+?)(?:-([a-z0-9]+))?$", exp)
    if m and m.group(1) in S1_ARMS:
        arm, suf = m.group(1), m.group(2) or ""
        if suf == "timed":
            return "1", "timed-orders", f"Stage-1 self-play in the timed-order env: {S1_ARMS[arm]}.", "used"
        objs = MORL_SET.get(suf, suf)
        pilot = suf == "smoke"
        return "1", "morl-benchmark", (
            f"Stage-1 self-play (MORL benchmark arm {arm}): {S1_ARMS[arm]}"
            + ("" if arm in ("bench_sp", "bench_sparse") and not suf else f"; {objs}")
            + ". 2e6 steps, 12 rollout threads; checkpoints init/mid/final become stage-2 partners."
        ), "pilot" if pilot else "used"
    m = re.match(r"^fcp-S2-(.+)$", exp)
    if m:
        rest = m.group(1)
        if rest.endswith("-pilot"):
            return "2", "morl-s2", f"Stage-2 pilot ({rest[:-6]} population).", "pilot"
        mm = re.match(r"^scripted-([a-z0-9]+?)(?:-(k\d|1e7))?$", rest)
        if mm:
            arm, extra = mm.group(1), mm.group(2)
            desc = f"Stage-2 trainee vs {SPECIALISTS}; {S2_REWARD.get(arm, arm)}."
            study = "fillin-specialists"
            if extra and extra.startswith("k"):
                desc = f"Partner removal: stage-2 trainee vs {extra[1:]} of the 5 scripted specialists (rotating subsets); {S2_REWARD.get(arm, arm)}."
                study = "partner-removal"
            elif extra == "1e7":
                desc += " Budget test at 1e7 steps (random3 did not train at 2e6)."
            return "2", study, desc, "used"
        if rest == "mixedspec-hand":
            return "2", "specialist-mix", "Stage-2 trainee vs the usual 18 learned partners + the 5 specialists (22% of games); hand-shaped reward.", "used"
        if rest == "mixspec50-hand":
            return "2", "specialist-mix", "Stage-2 trainee vs learned partners + specialists repeated to ~53% of games (18 + 5x4; random3 9 + 5x2); hand-shaped reward.", "used"
        if rest == "family16-hand":
            return "2", "specialist-family", "Stage-2 trainee vs 16 generated family specialists (family.py: task mix, noise, laziness, wandering) with swaps; no hand-written scripts; hand-shaped reward.", "used"
        if rest == "family16mix-hand":
            return "2", "specialist-family", "Stage-2 trainee vs the usual 18 learned partners + 16 family specialists (47% of games); hand-shaped reward.", "used"
        mm = re.match(r"^(bench_[a-z_]+?)(?:-([a-z0-9]+))?$", rest)
        if mm:
            pop, suf = mm.group(1), mm.group(2) or ""
            if pop == "bench_sp" and suf in ("rung1", "rung2", "rung3"):
                rung = {"rung1": "critic sees the partner id, actor does not (control)",
                        "rung2": "actor sees the raw scalar partner id",
                        "rung3": "actor sees a one-hot partner id"}[suf]
                return "2", "pid-ladder", f"Partner-ID ladder (oracle ceiling, not ZSC): usual population; {rung}.", "used"
            if pop == "bench_sp" and suf == "annego":
                return "2", "morl-s2", "Usual population (bench_sp stage-1 agents); the trainee's reward is annealed MORL ('MORL trainee', Finding 5).", "used"
            if pop == "bench_sp" and suf == "fillego":
                return "2", "morl-s2", "Usual population; the trainee's reward is the MORL fill/complement rule.", "used"
            if pop == "bench_sp" and not suf:
                return "2", "morl-s2", "Usual stage-2 (ZSC-Eval FCP): trainee vs the bench_sp population (seeds x init/mid/final); hand-shaped reward.", "used"
            return "2", "morl-s2", f"Stage-2 trainee (task reward) vs a population of {pop} stage-1 agents ({MORL_SET.get(suf, suf)}).", "used"
        if rest.startswith("mixed"):
            return "2", "morl-s2", "Stage-2 pilot: bench_sp + bench_morl_ad stage-1 agents mixed (reward-diversity question).", "pilot"
    return "?", "unknown", f"Unrecognised experiment '{exp}'.", "unreferenced"


def in_pool(layout, exp, pool):
    if exp == "hsp-s1":
        return f"{layout}/hsp/s1/hsp-s1" in pool
    if exp.startswith("fcp-S2-"):
        return f"{layout}/fcp/s2/{exp}" in pool
    return f"{layout}/fcp/s1/{exp}" in pool


def classify(run, pool, live=frozenset()):
    layout, exp, tags = run["layout"], run["exp"] or "", set(run["tags"])
    stage, study, desc, status = describe(layout, exp, tags)
    if run["state"] == "running":
        status = "running"
    elif (run["project"], layout, exp) in live and run["state"] == "finished":
        # Siblings still training: not extracted yet, so the pool cannot vouch for it.
        status = "in-progress"
    elif run["state"] in ("crashed", "failed", "killed"):
        status = "crashed"
    elif "superseded" in tags:
        status = "superseded"
        desc += " Superseded by the anchored-objective version of this arm."
    elif "unused" in tags or "hidden" in tags:
        status = "superseded"
        desc += " Excluded from extraction (duplicate seed or replaced run)."
    elif exp == "br" and layout == "unident_s" and "local" in tags:
        status = "superseded"
        desc += " Earlier set (workstation, Sep 7); BR-Prox in the October reports uses the Oct 3 set."
    elif exp.endswith("-1e7"):
        desc += " Read from its training curves (no extraction); cited in the 30 Sep report."
    elif status == "used" and stage in ("1", "2") and not in_pool(layout, exp, pool):
        status = "unreferenced"
        desc += " No checkpoints for this experiment in the cluster policy pool."
    return stage, study, desc, status


def plan(runs, pool):
    live = {(r["project"], r["layout"], r["exp"]) for r in runs if r["state"] == "running"}
    out = []
    for r in runs:
        stage, study, desc, status = classify(r, pool, live)
        host = r["notes"].split("host: ")[-1] if "host: " in r["notes"] else r["notes"]
        notes = f"{desc}\nKitchen: {r['layout']}. Seed {r['seed']}, {r['steps']} env steps.\nhost: {host}"
        new_tags = [f"stage-{stage}", f"study-{study}", f"kitchen-{r['layout']}", f"status-{status}"]
        tags = [t for t in r["tags"] if not t.startswith(("stage-", "study-", "kitchen-", "status-"))] + new_tags
        out.append(dict(r, status=status, study=study, stage=stage, new_notes=notes, new_tags=tags))
    return out


def dump(path):
    import wandb

    api = wandb.Api(timeout=120)
    ent = os.environ["WANDB_ENTITY"]
    with open(path, "w") as f:
        for p in api.projects(ent):
            for r in api.runs(f"{ent}/{p.name}", per_page=200):
                c = r.config
                f.write(json.dumps(dict(
                    project=p.name, id=r.id, name=r.name, state=r.state, created=r.created_at,
                    exp=c.get("experiment_name"), layout=c.get("layout_name"), seed=c.get("seed"),
                    steps=c.get("num_env_steps"), tags=list(r.tags), notes=r.notes or ""), default=str) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--dump", action="store_true", help="Refresh --inventory from the W&B API first")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--delete", nargs="*", default=[], metavar="STATUS")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--show", nargs="*", default=[], metavar="STATUS", help="List the runs with these statuses")
    args = ap.parse_args()
    if args.dump:
        dump(args.inventory)
    runs = [json.loads(line) for line in open(args.inventory)]
    pool = {line.strip().strip("./") for line in open(args.pool) if line.strip()}
    rows = plan(runs, pool)

    by = collections.Counter((r["status"]) for r in rows)
    print("status:", dict(by))
    print("study:", dict(collections.Counter(r["study"] for r in rows)))
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["status"], r["project"], r["layout"], r["exp"])].append(r)
    for st in args.show:
        print(f"\n== {st}")
        for k, v in sorted(groups.items()):
            if k[0] == st:
                print(f"  {k[1]:15s} {k[2]:14s} {k[3]:30s} {len(v):3d} runs   {v[0]['new_notes'].splitlines()[0][:90]}")
    unknown = [r for r in rows if r["study"] == "unknown"]
    if unknown:
        print("\nunrecognised:", collections.Counter((r["layout"], r["exp"]) for r in unknown))

    if not (args.apply or args.delete):
        return
    import wandb

    api = wandb.Api(timeout=120)
    ent = os.environ["WANDB_ENTITY"]
    if args.apply:
        n = 0
        for r in rows:
            if r["status"] in ("running",):
                continue
            if r["new_notes"] == r["notes"] and sorted(r["new_tags"]) == sorted(r["tags"]):
                continue
            run = api.run(f"{ent}/{r['project']}/{r['id']}")
            run.notes = r["new_notes"]
            run.tags = r["new_tags"]
            run.update()
            n += 1
            if n % 50 == 0:
                print(f"  annotated {n}", flush=True)
        print(f"annotated {n} runs")
    if args.delete:
        doomed = [r for r in rows if r["status"] in args.delete and r["status"] not in ("running", "in-progress")]
        for r in doomed:
            print(f"  delete {r['project']}/{r['id']} {r['layout']} {r['exp']} seed {r['seed']} ({r['state']})")
        if not doomed:
            return
        if not args.yes and input(f"Delete these {len(doomed)} runs? Type 'delete': ") != "delete":
            print("aborted")
            return
        for r in doomed:
            api.run(f"{ent}/{r['project']}/{r['id']}").delete()
        print(f"deleted {len(doomed)} runs")


if __name__ == "__main__":
    sys.exit(main())
