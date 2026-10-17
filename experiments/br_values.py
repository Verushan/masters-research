"""Best-response returns from the BR training logs, for br_prox.py.

pipelines/br-partners.slurm writes one log per held-out partner,
experiments/logs/br/{layout}/br_{key}.log, where the key is the HSP partner
index (12) or a self-play checkpoint (sp7_mid). Each log carries the periodic
stochastic evaluation `either-br_agent-eval_ep_sparse_r`; this keeps the last
one, the mean of the last two, and the best, per partner.

Run where the logs are (the cluster), then copy the JSON back:

    python experiments/br_values.py --layouts unident_s_cx pantry_cx \\
        --out experiments/results/br_values_cx.json
"""

import argparse
import glob
import json
import os.path as osp
import re

HERE = osp.dirname(osp.abspath(__file__))
EVAL = re.compile(r"'either-br_agent-eval_ep_sparse_r': '([0-9.]+)'")
STEPS = re.compile(r"total num timesteps (\d+)")


def parse(path):
    text = open(path).read()
    vals = [float(v) for v in EVAL.findall(text)]
    steps = STEPS.findall(text)
    return {
        "final": vals[-1] if vals else None,
        "last2": sum(vals[-2:]) / len(vals[-2:]) if vals else None,
        "best": max(vals) if vals else None,
        "n_evals": len(vals),
        "last_step": int(steps[-1]) if steps else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layouts", nargs="+", required=True)
    ap.add_argument("--logs", default=osp.join(HERE, "logs", "br"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {}
    for layout in args.layouts:
        out[layout] = {}
        for path in sorted(glob.glob(osp.join(args.logs, layout, "br_*.log"))):
            key = re.match(r"br_(.+)\.log$", osp.basename(path)).group(1)
            out[layout][key] = parse(path)
        done = [k for k, v in out[layout].items() if v["final"] is not None]
        print(f"{layout}: {len(done)}/{len(out[layout])} partners with an evaluation")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
