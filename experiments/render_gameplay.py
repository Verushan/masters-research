"""Side-by-side gameplay GIFs for the reports.

Each scene is two or more panels played from the same start, one game per
panel, drawn with the environment's own sprites (StateVisualizer). A seat is a
learned agent (a checkpoint in the policy pool) or a scripted cook (a key of
fillin_eval.PARTNERS).

The game shown is not hand-picked for drama: `--seeds N` plays N games per
panel first and renders the seed whose soup counts sit closest to each panel's
median, so the clip shows a typical game. The chosen seed and the medians are
printed and written next to the GIF.

    python experiments/render_gameplay.py --scene fillin_unident_s \\
        --out experiments/results/gifs/fillin_unident_s.gif
"""

import argparse
import json
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, osp.join(HERE, "..", "zsc-eval"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from fillin_eval import PARTNERS, load_actor  # noqa: E402

from zsceval.envs.overcooked.Overcooked_Env import Overcooked  # noqa: E402

FONT = osp.join(HERE, "..", "zsc-eval", "zsceval", "envs", "overcooked", "overcooked_ai_py", "data", "fonts",
                "Roboto-Regular.ttf")
RNN = "{layout}/policy_config/rnn_policy_config.pkl"
MLP = "{layout}/policy_config/mlp_policy_config.pkl"

# A seat: ("actor", config, checkpoint) or ("script", PARTNERS key).
# A panel: (title, layout, seat 0, seat 1, greedy).
USUAL = ("actor", RNN, "unident_s/fcp/s2/fcp-S2-bench_sp/1.pt")
A = ("actor", RNN, "unident_s/fcp/s2/fcp-S2-scripted-hand/1.pt")
SCENES = {
    "fillin_unident_s": dict(
        steps=200,
        note="Partner: a scripted potter, who only ever fills pots. Someone else has to plate and serve.",
        hats="blue hat: the agent · green hat: the potter",
        panels=[
            ("Usual stage-2 agent", "unident_s", USUAL, ("script", "potter"), False),
            ("Agent A (trained with specialists)", "unident_s", A, ("script", "potter"), False),
        ],
    ),
    "zeroshot_unident_s": dict(
        steps=200,
        note="Partner: held-out partner hsp10, a learned agent neither of them has met.",
        hats="blue hat: the agent · green hat: hsp10",
        panels=[
            ("Usual stage-2 agent", "unident_s", USUAL,
             ("actor", MLP, "unident_s/hsp/s1/hsp-s1/hsp10_final_w0_actor.pt"), False),
            ("Agent A (trained with specialists)", "unident_s", A,
             ("actor", MLP, "unident_s/hsp/s1/hsp-s1/hsp10_final_w0_actor.pt"), False),
        ],
    ),
    "scripts_random3": dict(
        steps=200,
        note="The same two scripted cooks (both do every job) in two kitchens.",
        panels=[
            ("unident_s", "unident_s", ("script", "generalist"), ("script", "generalist"), False),
            ("random3 (Counter Circuit)", "random3", ("script", "generalist"), ("script", "generalist"), False),
        ],
    ),
}


class Seat:
    def __init__(self, spec, layout):
        self.kind = spec[0]
        if self.kind == "actor":
            self.args, self.actor = load_actor(spec[1].format(layout=layout), spec[2], {})
        else:
            self.name = spec[1]

    def reset(self):
        if self.kind == "actor":
            self.rnn = np.zeros((1, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)

    def act(self, obs, avail, greedy, rng):
        with torch.no_grad():
            probs, rnn = self.actor.get_probs(np.asarray(obs, dtype=np.float32)[None], self.rnn,
                                              np.ones((1, 1), dtype=np.float32), np.asarray(avail)[None])
        self.rnn = rnn.cpu().numpy()
        p = probs.cpu().numpy().reshape(-1).astype(np.float64)
        p /= p.sum()
        return int(p.argmax()) if greedy else int(rng.choice(len(p), p=p))


def make_env(layout):
    import pickle

    with open(osp.join(os.environ["POLICY_POOL"], RNN.format(layout=layout)), "rb") as f:
        all_args = pickle.load(f)[0]
    all_args.layout_name = layout
    all_args.episode_length = 400
    env = Overcooked(all_args, run_dir=osp.join(os.environ.get("CLAUDE_JOB_DIR", HERE), "tmp"), evaluation=True)
    assert env.agent_idx == 0
    return env


def play(panel, seed, steps, keep_states):
    _title, layout, s0, s1, greedy = panel
    env = make_env(layout)
    seats = [Seat(s0, layout), Seat(s1, layout)]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs, _share, avail = env.reset()
    scripts = [None, None]
    for i, s in enumerate(seats):
        s.reset()
        if s.kind == "script":
            scripts[i] = PARTNERS[s.name]["make"]()
            scripts[i].reset(env.base_mdp, env.base_env.state, i)
    env.script_agent = scripts
    states, soups, total = [env.base_env.state.deepcopy()], [0], 0
    for _t in range(steps):
        joint = [[s.act(obs[i], avail[i], greedy, rng)] if s.kind == "actor" else [4] for i, s in enumerate(seats)]
        obs, _share, _r, _dones, info, avail = env.step(np.array(joint))
        total += int(round(sum(info["sparse_r_by_agent"]) / 20))
        soups.append(total)
        if keep_states:
            states.append(env.base_env.state.deepcopy())
    return env.base_mdp, states, soups


def render(scene, results, out, seed):
    from zsceval.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer

    vis = StateVisualizer(tile_size=48, is_rendering_hud=False, is_rendering_cooking_timer=True)
    title_font, body_font, small = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 16), ImageFont.truetype(FONT, 14)
    pad, head, foot = 16, 58, 30
    frames_by_panel = []
    for mdp, states, _ in results:
        frames = []
        for st in states:
            surf = vis.render_state(st, grid=mdp.terrain_mtx)
            import pygame

            frames.append(Image.fromarray(pygame.surfarray.array3d(surf).transpose(1, 0, 2)))
        frames_by_panel.append(frames)
    widths = [f[0].width for f in frames_by_panel]
    height = max(f[0].height for f in frames_by_panel)
    W = sum(widths) + pad * (len(widths) + 1)
    H = head + height + foot + pad
    out_frames = []
    n = len(frames_by_panel[0])
    for t in range(n):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        x = pad
        for (panel, frames, (_m, _s, soups)) in zip(scene["panels"], frames_by_panel, results):
            d.text((x, 10), panel[0], font=title_font, fill=(20, 20, 20))
            d.text((x, 34), f"soups served: {soups[t]}", font=body_font, fill=(37, 99, 235))
            if scene.get("hats"):
                d.text((x + 150, 36), scene["hats"], font=small, fill=(90, 90, 90))
            canvas.paste(frames[t], (x, head))
            x += frames[t].width + pad
        d.text((pad, head + height + 8), f"step {t} of {n - 1}  ·  {scene['note']}", font=small, fill=(90, 90, 90))
        out_frames.append(canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    hold = [out_frames[-1]] * 15
    out_frames[0].save(out, save_all=True, append_images=out_frames[1:] + hold, duration=100, loop=0, optimize=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", choices=sorted(SCENES), required=True)
    ap.add_argument("--seeds", type=int, default=10, help="Games per panel to pick a typical seed from")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    scene = SCENES[args.scene]

    counts = np.array([[play(p, s, scene["steps"], False)[2][-1] for s in range(args.seeds)] for p in scene["panels"]])
    med = np.median(counts, axis=1, keepdims=True)
    dev = (np.abs(counts - med) / (counts.std(axis=1, keepdims=True) + 1)).sum(0)
    seed = int(dev.argmin())
    print(f"soups per panel over {args.seeds} seeds:\n{counts}\nmedians {med.ravel()} -> seed {seed}: {counts[:, seed]}")

    results = [play(p, seed, scene["steps"], True) for p in scene["panels"]]
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    render(scene, results, args.out, seed)
    meta = {"scene": args.scene, "seed": seed, "steps": scene["steps"],
            "panels": [p[0] for p in scene["panels"]], "soups_by_seed": counts.tolist(),
            "medians": med.ravel().tolist(), "shown": counts[:, seed].tolist()}
    with open(osp.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {args.out} ({osp.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
