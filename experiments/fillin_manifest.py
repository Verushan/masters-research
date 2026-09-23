"""List every trained agent the fill-in suite should score, with how to load it.

One tab-separated line per agent: name, actor, policy config, env flags. The
config and flags follow from the arm, because three kinds of agent need three
kinds of loading:

    plain           mlp config (stage 1) / rnn config (stage 2), no flags
    partner picture (…_shares…)  the config built with --use_morl_obs_shares
    fill-in         (…fill…)     the config with preference weights and partner
                    picture, plus the MORL settings it trained under, so its
                    priorities move against each scripted partner as in training

Scanning the pool rather than listing seeds means an arm that lost a seed is
scored on what exists instead of failing on what doesn't.

    python experiments/fillin_manifest.py unident_s > manifest.tsv
"""

import argparse
import os
import os.path as osp
import re

S1_ARMS = [
    "bench_sp",
    "bench_sparse",
    "bench_morl-live3",
    "bench_morl_ad-live3",
    "bench_morl_ann-live3t",
    "bench_morl_fill-live3t",
    "bench_sp_shares-live3",
]
S2_EXPS = [
    "fcp-S2-bench_sp",
    "fcp-S2-bench_sp-annego",
    "fcp-S2-bench_sp-fillego",
    "fcp-S2-bench_morl-live3",
    "fcp-S2-bench_morl_ad-live3",
]


def objective_set(layout):
    # unident_s has no counter handoffs, so its arms trained on the
    # three-objective set; every other layout on the four-objective one.
    return ("anchored_live3", "20,3,3") if layout == "unident_s" else ("anchored", "20,3,3,3")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("layout")
    ap.add_argument("--s1_arms", nargs="*", default=S1_ARMS)
    ap.add_argument("--s2_exps", nargs="*", default=S2_EXPS)
    args = ap.parse_args()

    pool = os.environ.get("POLICY_POOL")
    assert pool, "POLICY_POOL is unset"
    obj, ann_w = objective_set(args.layout)
    cfg = f"{args.layout}/policy_config"
    fill_flags = (
        f"morl_weights={ann_w} morl_anneal_dense=true morl_team_task=true "
        "morl_adaptive_weights=true morl_adaptive_target=complement"
    )

    def emit(name, actor, config, flags=""):
        print("\t".join([name, actor, config, flags]))

    for arm in args.s1_arms:
        d = osp.join(pool, args.layout, "fcp", "s1", arm)
        if not osp.isdir(d):
            continue
        seeds = sorted(int(m.group(1)) for f in os.listdir(d) if (m := re.match(r"sp(\d+)_final_actor\.pt$", f)))
        for s in seeds:
            actor = f"{args.layout}/fcp/s1/{arm}/sp{s}_final_actor.pt"
            if "fill" in arm:
                emit(f"{arm}_s{s}", actor, f"{cfg}/mlp_policy_config_mow-{obj}_mos.pkl", fill_flags)
            elif "shares" in arm:
                emit(f"{arm}_s{s}", actor, f"{cfg}/mlp_policy_config_mos-{obj}.pkl")
            else:
                emit(f"{arm}_s{s}", actor, f"{cfg}/mlp_policy_config.pkl")

    for exp in args.s2_exps:
        d = osp.join(pool, args.layout, "fcp", "s2", exp)
        if not osp.isdir(d):
            continue
        seeds = sorted(int(m.group(1)) for f in os.listdir(d) if (m := re.match(r"(\d+)\.pt$", f)))
        arm = "s2_" + exp[len("fcp-S2-"):]
        for s in seeds:
            actor = f"{args.layout}/fcp/s2/{exp}/{s}.pt"
            if "fillego" in exp:
                emit(f"{arm}_s{s}", actor, f"{cfg}/rnn_policy_config_mow-{obj}_mos.pkl", fill_flags)
            else:
                emit(f"{arm}_s{s}", actor, f"{cfg}/rnn_policy_config.pkl")


if __name__ == "__main__":
    main()
