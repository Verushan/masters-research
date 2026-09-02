#!/usr/bin/env python
"""Partner-conditioned analysis of a cross-play matrix (workstream A).

`analyze_crossplay.py` collapses the matrix to one number per agent -- the ZSC
mean -- and that number is what currently ranks the arms. This script keeps the
matrix and asks what the mean throws away:

  A1  variance decomposition   how much of the return variance is the ego agent,
                               how much is *which partner*, how much is the
                               specific pairing
  A2  partner personalities    cluster the held-out partners by how they behave,
                               then report each arm per cluster rather than per
                               partner
  A3  adaptation index         how much an ego agent changes its *own* behaviour
                               as a function of who it is paired with
  A4  does adaptation pay      within-arm correlation between adapting to a
                               partner and scoring against them
  A5  regret and CVaR          br_hat - return per pairing, and the mean over the
                               worst quartile of partners
  D3  paired tests             arm-vs-arm over the shared partner set, which is
                               a paired comparison and should be tested as one

A1/A2/A5/D3 read the metrics JSON that `analyze_crossplay.py` already writes.
A3/A4 need per-episode behaviour, so they read the cross-play records directly.

Usage:
    python experiments/analyze_partner_effects.py \
        --metrics results/metrics_unident_s_hsp_n6.json \
        --records results/cross_play_unident_s_hsp_deterministic.json.gz \
        --out results/partner_effects_unident_s.json
"""

import argparse
import gzip
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Event counters used to describe "how an agent behaved this episode". Shared by
# the adaptation index (A3) and the partner clustering (A2) so the two are
# expressed in the same space. Deliberately excludes the sparse/shaped reward --
# this is a description of behaviour, not of outcome.
BEHAVIOUR_EVENTS = [
    "pickup_onion_from_O",
    "pickup_onion_from_X",
    "put_onion_on_X",
    "PLACEMENT_IN_POT",
    "pickup_dish_from_D",
    "put_dish_on_X",
    "USEFUL_DISH_PICKUP",
    "SOUP_PICKUP",
    "put_soup_on_X",
    "pickup_soup_from_X",
    "delivery",
    "STAY",
    "MOVEMENT",
]

RNG_SEED = 0
N_BOOTSTRAP = 2000


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def ego_agents(metrics):
    """Pool entries that are being evaluated, not the partners they face.

    `_peak` checkpoints are a second checkpoint of an agent already in the pool,
    so including them would double-count that agent's seed in every per-arm
    statistic.
    """
    return [
        name
        for name, entry in metrics["per_agent"].items()
        if not entry["group"].startswith("heldout") and not name.endswith("_peak")
    ]


def return_matrix(metrics):
    """(ego, partner) return matrix plus the row/column labels."""
    egos = ego_agents(metrics)
    partners = metrics["partners"]
    matrix = np.array(
        [[metrics["per_agent"][e]["vs_partners"][p] for p in partners] for e in egos],
        dtype=np.float64,
    )
    return matrix, egos, partners


# ---------------------------------------------------------------------------
# A1 -- variance decomposition
# ---------------------------------------------------------------------------


def decompose_variance(matrix):
    """Two-way split of the return variance into ego / partner / interaction.

    The additive model is `r_ij = mu + a_i + b_j + e_ij` fitted by the usual
    marginal means, so the three sums of squares partition the total exactly.
    Reported as fractions of the total, which is what makes "the ZSC mean ranks
    on the smallest term" a statement about this matrix rather than a slogan.
    """
    grand = matrix.mean()
    ego_effect = matrix.mean(axis=1) - grand
    partner_effect = matrix.mean(axis=0) - grand
    residual = matrix - grand - ego_effect[:, None] - partner_effect[None, :]

    n_ego, n_partner = matrix.shape
    ss_total = float(((matrix - grand) ** 2).sum())
    ss_ego = float((ego_effect**2).sum() * n_partner)
    ss_partner = float((partner_effect**2).sum() * n_ego)
    ss_interaction = float((residual**2).sum())

    return {
        "grand_mean": float(grand),
        "n_ego": int(n_ego),
        "n_partner": int(n_partner),
        "ss_total": ss_total,
        "frac_ego": ss_ego / ss_total if ss_total else 0.0,
        "frac_partner": ss_partner / ss_total if ss_total else 0.0,
        "frac_interaction": ss_interaction / ss_total if ss_total else 0.0,
        "partner_marginals": {
            "min": float(partner_effect.min() + grand),
            "max": float(partner_effect.max() + grand),
            "std": float(partner_effect.std()),
        },
        "ego_marginals": {
            "min": float(ego_effect.min() + grand),
            "max": float(ego_effect.max() + grand),
            "std": float(ego_effect.std()),
        },
    }


def bootstrap_variance_fractions(matrix, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Percentile CIs for the three fractions, resampling both margins.

    Ego rows and partner columns are both resampled, because both are samples --
    seeds from a training distribution and partners from the HSP candidate
    space. Resampling only one would understate the uncertainty on the other's
    variance component.
    """
    rng = np.random.default_rng(seed)
    n_ego, n_partner = matrix.shape
    draws = defaultdict(list)

    for _ in range(n_boot):
        rows = rng.integers(0, n_ego, n_ego)
        cols = rng.integers(0, n_partner, n_partner)
        sample = matrix[np.ix_(rows, cols)]
        if sample.std() == 0:
            continue
        split = decompose_variance(sample)
        for key in ("frac_ego", "frac_partner", "frac_interaction"):
            draws[key].append(split[key])

    return {
        key: {
            "lo": float(np.percentile(values, 2.5)),
            "hi": float(np.percentile(values, 97.5)),
        }
        for key, values in draws.items()
        if values
    }


# ---------------------------------------------------------------------------
# behaviour profiles -- shared by A2 and A3
# ---------------------------------------------------------------------------


def episode_profile(record, side):
    """L1-normalised event profile for one player in one episode.

    Normalising makes this a description of *how* the agent spent the episode
    rather than how much it did, so a slow agent and a fast agent running the
    same protocol land in the same place.
    """
    counts = np.array(
        [record.get(f"eval_ep_{event}_by_agent{side}", 0.0) for event in BEHAVIOUR_EVENTS],
        dtype=np.float64,
    )
    total = counts.sum()
    return counts / total if total > 0 else counts


def collect_profiles(records, partners):
    """Per-(ego, partner) mean behaviour profiles, for the ego and the partner.

    Both seating orders are folded together: `cross_play.py` writes agent0/agent1
    as the pool ordering, not as roles, so a pairing appears in either order and
    both are the same pairing.
    """
    partner_set = set(partners)
    ego_profiles = defaultdict(lambda: defaultdict(list))
    partner_profiles = defaultdict(list)

    for record in records:
        pairing = ((record["agent0"], record["agent1"], 0, 1), (record["agent1"], record["agent0"], 1, 0))
        for ego, partner, ego_side, partner_side in pairing:
            if partner not in partner_set or ego in partner_set:
                continue
            ego_profiles[ego][partner].append(episode_profile(record, ego_side))
            partner_profiles[partner].append(episode_profile(record, partner_side))

    ego_mean = {
        ego: {partner: np.mean(rows, axis=0) for partner, rows in by_partner.items()}
        for ego, by_partner in ego_profiles.items()
    }
    partner_mean = {
        partner: np.mean(rows, axis=0) for partner, rows in partner_profiles.items()
    }
    return ego_mean, partner_mean


# ---------------------------------------------------------------------------
# A3 -- adaptation index
# ---------------------------------------------------------------------------


def adaptation_index(ego_mean):
    """Mean distance of an ego's per-partner profile from its own average.

    Zero means the agent runs one protocol regardless of partner; larger means
    its behaviour is a function of who it is with. This is the direct
    behavioural reading of "does the agent adapt to its partner", and it is
    independent of whether the adaptation helps -- that is A4.
    """
    scores = {}
    for ego, by_partner in ego_mean.items():
        if len(by_partner) < 2:
            continue
        profiles = np.array(list(by_partner.values()))
        centre = profiles.mean(axis=0)
        scores[ego] = float(np.linalg.norm(profiles - centre, axis=1).mean())
    return scores


# ---------------------------------------------------------------------------
# A2 -- partner personalities
# ---------------------------------------------------------------------------


def cluster_partners(partner_mean, partners, n_clusters=4, seed=RNG_SEED):
    """k-means over partner behaviour profiles, labelled by dominant behaviour.

    Clustering on observed behaviour rather than on the `w0` reward vector is
    deliberate: two bias vectors can induce the same policy, and it is the
    policy the ego agent actually has to coordinate with. The label is the event
    the cluster centre over-weights most relative to the population, which is
    what makes a cluster nameable ("stays put", "hoards onions").
    """
    usable = [p for p in partners if p in partner_mean]
    if len(usable) < n_clusters:
        return {}, {}

    features = np.array([partner_mean[p] for p in usable])
    rng = np.random.default_rng(seed)

    # k-means++ style init, then Lloyd iterations. Small enough (16 x 13) that a
    # dependency on scikit-learn is not worth taking for this.
    centres = features[rng.choice(len(features), 1)]
    while len(centres) < n_clusters:
        distances = np.min(((features[:, None] - centres[None]) ** 2).sum(-1), axis=1)
        if distances.sum() == 0:
            break
        centres = np.vstack([centres, features[rng.choice(len(features), p=distances / distances.sum())]])

    labels = np.zeros(len(features), dtype=int)
    for _ in range(100):
        new_labels = np.argmin(((features[:, None] - centres[None]) ** 2).sum(-1), axis=1)
        if (new_labels == labels).all():
            break
        labels = new_labels
        for k in range(len(centres)):
            if (labels == k).any():
                centres[k] = features[labels == k].mean(axis=0)

    population_mean = features.mean(axis=0)
    assignment, descriptions = {}, {}
    for k in range(len(centres)):
        members = [usable[i] for i in range(len(usable)) if labels[i] == k]
        if not members:
            continue
        lift = centres[k] - population_mean
        dominant = BEHAVIOUR_EVENTS[int(np.argmax(lift))]
        name = f"c{k}_{dominant}"
        descriptions[name] = {
            "members": members,
            "size": len(members),
            "dominant_event": dominant,
            "profile": {e: float(v) for e, v in zip(BEHAVIOUR_EVENTS, centres[k])},
        }
        for member in members:
            assignment[member] = name
    return assignment, descriptions


# ---------------------------------------------------------------------------
# A5 / D3
# ---------------------------------------------------------------------------


def regret_and_cvar(metrics, matrix, egos, partners, worst_fraction=0.25):
    """Per-ego regret against br_hat, plus the mean over the worst quartile.

    `br_hat` is the optimistic best-response proxy `analyze_crossplay.py`
    already stores. Regret is the more honest statistic than raw return: a
    partner nobody can score against should not count against an arm, and a
    partner everybody else exploits should.
    """
    br_hat = np.array([metrics["br_hat"][p] for p in partners], dtype=np.float64)
    k = max(1, int(round(worst_fraction * len(partners))))

    rows = {}
    for i, ego in enumerate(egos):
        returns = matrix[i]
        regret = np.maximum(br_hat - returns, 0.0)
        achievable = br_hat > 0
        rows[ego] = {
            "group": metrics["per_agent"][ego]["group"],
            "mean_regret": float(regret.mean()),
            "mean_regret_ratio": (
                float((returns[achievable] / br_hat[achievable]).mean()) if achievable.any() else 0.0
            ),
            "cvar_worst": float(np.sort(returns)[:k].mean()),
            "worst_partners": [partners[j] for j in np.argsort(returns)[:k]],
        }
    return rows


def group_rows(metrics, egos):
    grouped = defaultdict(list)
    for i, ego in enumerate(egos):
        grouped[metrics["per_agent"][ego]["group"]].append(i)
    return grouped


def paired_tests(matrix, egos, metrics):
    """Arm vs arm, paired over the shared partner set.

    Every arm meets the same partners, so the comparison is paired and an
    unpaired test would throw away the partner-identity term that A1 shows is
    the dominant one. Falls back to a sign test when scipy is unavailable.
    """
    grouped = group_rows(metrics, egos)
    profiles = {g: matrix[idx].mean(axis=0) for g, idx in grouped.items()}

    try:
        from scipy import stats
    except ImportError:
        stats = None

    results = []
    for a, b in itertools.combinations(sorted(profiles), 2):
        diff = profiles[a] - profiles[b]
        row = {
            "arm_a": a,
            "arm_b": b,
            "mean_diff": float(diff.mean()),
            "sd_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
            "a_better_on": int((diff > 0).sum()),
            "n_partners": int(len(diff)),
        }
        if stats is not None and diff.std() > 0:
            row["t_p"] = float(stats.ttest_rel(profiles[a], profiles[b]).pvalue)
            row["wilcoxon_p"] = float(stats.wilcoxon(profiles[a], profiles[b]).pvalue)
        results.append(row)
    return results


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def summarise(values):
    array = np.array(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "n": int(len(array)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", required=True, help="metrics_*.json from analyze_crossplay.py")
    parser.add_argument("--records", nargs="*", default=[], help="cross_play_*.json[.gz] for the behaviour metrics")
    parser.add_argument("--out", required=True)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    metrics = load_json(args.metrics)
    matrix, egos, partners = return_matrix(metrics)
    grouped = group_rows(metrics, egos)

    report = {
        "source_metrics": str(args.metrics),
        "partner_group": metrics.get("partner_group"),
        "variance_decomposition": decompose_variance(matrix),
        "variance_ci": bootstrap_variance_fractions(matrix, n_boot=args.bootstrap),
        "paired_tests": paired_tests(matrix, egos, metrics),
    }

    regret = regret_and_cvar(metrics, matrix, egos, partners)
    report["regret_per_agent"] = regret
    report["regret_per_group"] = {
        group: {
            "mean_regret": summarise([regret[egos[i]]["mean_regret"] for i in idx]),
            "mean_regret_ratio": summarise([regret[egos[i]]["mean_regret_ratio"] for i in idx]),
            "cvar_worst": summarise([regret[egos[i]]["cvar_worst"] for i in idx]),
        }
        for group, idx in grouped.items()
    }

    if args.records:
        records = []
        for path in args.records:
            records.extend(load_json(path)["records"])
        ego_mean, partner_mean = collect_profiles(records, partners)

        adaptation = adaptation_index(ego_mean)
        report["adaptation_per_agent"] = adaptation
        report["adaptation_per_group"] = {
            group: summarise([adaptation[egos[i]] for i in idx if egos[i] in adaptation])
            for group, idx in grouped.items()
            if any(egos[i] in adaptation for i in idx)
        }

        assignment, clusters = cluster_partners(partner_mean, partners, n_clusters=args.clusters)
        report["partner_clusters"] = clusters
        if assignment:
            per_cluster = defaultdict(lambda: defaultdict(list))
            for i, ego in enumerate(egos):
                group = metrics["per_agent"][ego]["group"]
                for j, partner in enumerate(partners):
                    if partner in assignment:
                        per_cluster[group][assignment[partner]].append(matrix[i, j])
            report["return_by_cluster"] = {
                group: {cluster: summarise(values) for cluster, values in by_cluster.items()}
                for group, by_cluster in per_cluster.items()
            }

        # A4: within an arm, does adapting to a partner correlate with scoring
        # against them? Computed per (ego, partner) so it is a pairing-level
        # question, not an agent-level one.
        pairing = defaultdict(lambda: ([], []))
        for i, ego in enumerate(egos):
            if ego not in ego_mean:
                continue
            group = metrics["per_agent"][ego]["group"]
            centre = np.mean(list(ego_mean[ego].values()), axis=0)
            for j, partner in enumerate(partners):
                if partner in ego_mean[ego]:
                    pairing[group][0].append(float(np.linalg.norm(ego_mean[ego][partner] - centre)))
                    pairing[group][1].append(matrix[i, j])
        report["adaptation_pays"] = {
            group: {
                "pearson_r": float(np.corrcoef(dev, ret)[0, 1]),
                "n_pairings": len(dev),
            }
            for group, (dev, ret) in pairing.items()
            if len(dev) > 2 and np.std(dev) > 0 and np.std(ret) > 0
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))

    # -- console summary ----------------------------------------------------
    split = report["variance_decomposition"]
    ci = report.get("variance_ci", {})

    def band(key):
        if key not in ci:
            return ""
        return f"  [{100 * ci[key]['lo']:.0f}-{100 * ci[key]['hi']:.0f}]"

    print(f"\n{args.metrics}  ({split['n_ego']} ego x {split['n_partner']} partners, mean {split['grand_mean']:.1f})")
    print("  variance explained")
    print(f"    ego agent    {100 * split['frac_ego']:5.1f}%{band('frac_ego')}")
    print(f"    partner      {100 * split['frac_partner']:5.1f}%{band('frac_partner')}")
    print(f"    interaction  {100 * split['frac_interaction']:5.1f}%{band('frac_interaction')}")
    print(
        f"    partner marginals {split['partner_marginals']['min']:.0f} .. {split['partner_marginals']['max']:.0f}"
    )

    if "adaptation_per_group" in report:
        print("  adaptation index (higher = ego behaviour depends more on partner)")
        for group, stat in sorted(report["adaptation_per_group"].items(), key=lambda kv: -kv[1]["mean"]):
            print(f"    {group:22s} {stat['mean']:.3f} +- {stat['std']:.3f}  (n={stat['n']})")

    print("  regret vs br_hat / CVaR over worst quartile of partners")
    for group, stat in sorted(report["regret_per_group"].items(), key=lambda kv: kv[1]["mean_regret"]["mean"]):
        print(
            f"    {group:22s} regret {stat['mean_regret']['mean']:6.1f}"
            f"   captured {100 * stat['mean_regret_ratio']['mean']:5.1f}%"
            f"   cvar {stat['cvar_worst']['mean']:6.1f}"
        )

    if report.get("return_by_cluster"):
        clusters = sorted({c for by in report["return_by_cluster"].values() for c in by})
        print("  return by partner personality")
        print("    " + " " * 22 + "".join(f"{c[:18]:>20s}" for c in clusters))
        for group in sorted(report["return_by_cluster"]):
            cells = "".join(
                f"{report['return_by_cluster'][group].get(c, {'mean': float('nan')})['mean']:20.1f}" for c in clusters
            )
            print(f"    {group:22s}{cells}")

    significant = [r for r in report["paired_tests"] if r.get("wilcoxon_p", 1.0) < 0.05]
    print(f"  paired arm comparisons: {len(significant)}/{len(report['paired_tests'])} significant at p<0.05")
    for row in sorted(significant, key=lambda r: r["wilcoxon_p"]):
        print(
            f"    {row['arm_a']:22s} - {row['arm_b']:22s} {row['mean_diff']:+7.1f}"
            f"   {row['a_better_on']}/{row['n_partners']} partners   p={row['wilcoxon_p']:.3f}"
        )
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
