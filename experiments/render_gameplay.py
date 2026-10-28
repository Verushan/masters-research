"""Side-by-side gameplay GIFs for the reports.

Each scene is two or more panels played from the same start, one game per
panel, drawn with the environment's own sprites (StateVisualizer). A seat is a
learned agent (a checkpoint in the policy pool) or a scripted cook (a key of
fillin_eval.PARTNERS, or on a multi-recipe layout of NEW_PARTNERS or of that
env's SCRIPT_AGENTS, e.g. `dish_OOOTT`).

Multi-recipe layouts (`*_cx`) run in the new env and are drawn with its sprites.
Those only cover pots of up to three ingredients, so a bigger soup is drawn as
the nearest three-ingredient one and every soup is labelled with its contents
("3o 2t"); the score line counts dishes and points, since a dish is worth 10-60.

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

from fillin_eval import PARTNERS, kitchen, load_actor  # noqa: E402

from zsceval.overcooked_config import OLD_LAYOUTS  # noqa: E402

FONT = osp.join(HERE, "..", "zsc-eval", "zsceval", "envs", "overcooked", "overcooked_ai_py", "data", "fonts",
                "Roboto-Regular.ttf")
RNN = "{layout}/policy_config/rnn_policy_config.pkl"
MLP = "{layout}/policy_config/mlp_policy_config.pkl"

# A seat: ("actor", config, checkpoint) or ("script", PARTNERS key).
# A panel: (title, layout, seat 0, seat 1, greedy).
USUAL = ("actor", RNN, "unident_s/fcp/s2/fcp-S2-bench_sp/1.pt")
A = ("actor", RNN, "unident_s/fcp/s2/fcp-S2-scripted-hand/1.pt")
A1 = ("actor", RNN, "random1/fcp/s2/fcp-S2-scripted-hand/2.pt")
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
    "specialists_unident_s": dict(
        steps=150,
        tile=32,
        cols=3,
        note="Blue hat: agent A. Green hat: the specialist. Each panel is a separate game.",
        panels=[
            ("potter: only fills pots", "unident_s", A, ("script", "potter"), False),
            ("server: only plates and serves", "unident_s", A, ("script", "server"), False),
            ("idle: does nothing", "unident_s", A, ("script", "idle"), False),
            ("dial8: everything, prefers onions", "unident_s", A, ("script", "dial8"), False),
            ("dial2: everything, prefers serving", "unident_s", A, ("script", "dial2"), False),
        ],
    ),
    "greedy_random1": dict(
        steps=400,
        tile=64,
        note="Two copies of agent A (run 2). Greedy: always its most likely move.",
        panels=[
            ("Greedy actions", "random1", A1, A1, True),
            ("Sampled actions (ZSC-Eval)", "random1", A1, A1, False),
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


# Complex-dish kitchens (21-28 October reports). Typical runs, not the best:
# each agent is the run whose score is nearest its arm's mean. The HSP partner
# is hsp7: its gap between the two agents (+18) is near the mean gap (+21) and
# both agents cook with it (hsp40, nearer still, gave both 0 within a clip).
CX_USUAL_OPEN = ("actor", RNN, "unident_s_cx/fcp/s2/fcp-S2-usual-hand-v2/10.pt")
CX_SPEC_OPEN = ("actor", RNN, "unident_s_cx/fcp/s2/fcp-S2-partial-hand-v2/10.pt")
CX_USUAL_PANTRY = ("actor", RNN, "pantry_cx/fcp/s2/fcp-S2-usual-hand-v2/6.pt")
CX_SPECMORL_PANTRY = ("actor", RNN, "pantry_cx/fcp/s2/fcp-S2-scripted-dishann-v2/10.pt")
CX_HSP7 = ("actor", MLP, "pantry_cx/hsp/s1/hsp-s1/hsp7_final_w0_actor.pt")
BEST_DISH = ("script", "dish_OOOTT")
SCENES.update({
    "kitchens_cx": dict(
        steps=200,
        tile=40,
        cols=2,
        note="Two scripted cooks that make the 60-point dish (3 onions + 2 tomatoes) in every pot.",
        panels=[
            ("Open kitchen (unident_s_cx)", "unident_s_cx", BEST_DISH, BEST_DISH, False),
            ("Split pantry (pantry_cx)", "pantry_cx", BEST_DISH, BEST_DISH, False),
            ("Doorway (doorway_cx)", "doorway_cx", BEST_DISH, BEST_DISH, False),
            ("Big kitchen (bigkitchen_cx)", "bigkitchen_cx", BEST_DISH, BEST_DISH, False),
        ],
    ),
    "zeroshot_pantry_cx": dict(
        steps=400,
        tile=56,
        note="Partner: HSP partner hsp7, a held-out agent neither has met. Onions are on the left, tomatoes on the right.",
        hats="blue hat: the agent · green hat: hsp7",
        panels=[
            ("Usual stage-2 agent", "pantry_cx", CX_USUAL_PANTRY, CX_HSP7, False),
            ("Specialists + MORL agent", "pantry_cx", CX_SPECMORL_PANTRY, CX_HSP7, False),
        ],
    ),
    "fillin_unident_s_cx": dict(
        steps=250,
        tile=56,
        note="Partner: a scripted potter, who only fills pots. The agent has to plate and serve.",
        hats="blue hat: the agent · green hat: the potter",
        panels=[
            ("Usual stage-2 agent", "unident_s_cx", CX_USUAL_OPEN, ("script", "potter"), False),
            ("Agent trained with partial specialists", "unident_s_cx", CX_SPEC_OPEN, ("script", "potter"), False),
        ],
    ),
})


def is_new(layout):
    return layout not in OLD_LAYOUTS


def make_script(name, layout):
    partners = kitchen(layout)[1]
    if name in partners:
        return partners[name]["make"]()
    from zsceval.envs.overcooked_new.script_agent.script_agent import SCRIPT_AGENTS as NEW

    return NEW[name]()


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
    all_args.old_dynamics = not is_new(layout)
    Overcooked = kitchen(layout)[0]
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
            scripts[i] = make_script(s.name, layout) if is_new(layout) else PARTNERS[s.name]["make"]()
            scripts[i].reset(env.base_mdp, env.base_env.state, i)
    env.script_agent = scripts
    # soups: dishes served so far; points: their value (the two differ on the
    # multi-recipe menus, where a dish is worth 10-60).
    states, soups, points, total, pts = [env.base_env.state.deepcopy()], [0], [0], 0, 0
    for _t in range(steps):
        joint = [[s.act(obs[i], avail[i], greedy, rng)] if s.kind == "actor" else [4] for i, s in enumerate(seats)]
        obs, _share, _r, _dones, info, avail = env.step(np.array(joint))
        r = [float(x) for x in info["sparse_r_by_agent"]]
        total += sum(1 for x in r if x > 0) if is_new(layout) else int(round(sum(r) / 20))
        pts += sum(r)
        soups.append(total)
        points.append(int(pts))
        if keep_states:
            states.append(env.base_env.state.deepcopy())
    return env.base_mdp, states, soups, points, layout


def visualizer(layout, tile):
    if not is_new(layout):
        from zsceval.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer

        return StateVisualizer(tile_size=tile, is_rendering_hud=False, is_rendering_cooking_timer=True)
    from zsceval.envs.overcooked_new.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer

    if not getattr(StateVisualizer, "_capped", False):
        orig = StateVisualizer._soup_frame_name

        def capped(ingredients, status):
            o, t = ingredients.count("onion"), ingredients.count("tomato")
            if o + t > 3:  # no sprite: the nearest three-ingredient mix, labelled below
                o = round(3 * o / (o + t))
                t = 3 - o
            return orig(["onion"] * o + ["tomato"] * t, status)

        StateVisualizer._soup_frame_name = staticmethod(capped)
        StateVisualizer._capped = True
    return StateVisualizer(tile_size=tile, is_rendering_hud=False, is_rendering_cooking_timer=True)


def label_soups(img, state, tile, font):
    """Write each soup's true contents on its tile ("3o 2t")."""
    d = ImageDraw.Draw(img)
    for obj in state.objects.values():
        if obj.name != "soup":
            continue
        o, t = obj.ingredients.count("onion"), obj.ingredients.count("tomato")
        text = " ".join(x for x in (f"{o}o" if o else "", f"{t}t" if t else "") if x)
        x, y = obj.position[0] * tile, obj.position[1] * tile
        w = d.textlength(text, font=font)
        d.rectangle([x + 1, y + tile - 17, x + w + 7, y + tile - 1], fill=(255, 255, 255))
        d.text((x + 4, y + tile - 17), text, font=font, fill=(20, 20, 20))


def render(scene, results, out, seed):
    import pygame

    tile = scene.get("tile", 48)
    title_font, body_font, small = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 16), ImageFont.truetype(FONT, 14)
    pad, head, foot = 16, 58, 30
    frames_by_panel = []
    for mdp, states, _soups, _pts, layout in results:
        vis = visualizer(layout, tile)
        frames = []
        for st in states:
            surf = vis.render_state(st, grid=mdp.terrain_mtx)
            img = Image.fromarray(pygame.surfarray.array3d(surf).transpose(1, 0, 2))
            if is_new(layout):
                label_soups(img, st, tile, small)
            frames.append(img)
        frames_by_panel.append(frames)
    # Panels sit in a grid of `cols` columns, each cell sized to the largest panel.
    cols = scene.get("cols", len(frames_by_panel))
    rows = -(-len(frames_by_panel) // cols)
    cw = max(f[0].width for f in frames_by_panel)
    ch = head + max(f[0].height for f in frames_by_panel)
    W = cols * cw + pad * (cols + 1)
    H = rows * ch + pad * (rows - 1) + foot + pad
    out_frames = []
    n = len(frames_by_panel[0])
    for t in range(n):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        for i, (panel, frames, (_m, _s, soups, pts, layout)) in enumerate(zip(scene["panels"], frames_by_panel, results)):
            x, y = pad + (i % cols) * (cw + pad), (i // cols) * (ch + pad)
            d.text((x, y + 10), panel[0], font=title_font, fill=(20, 20, 20))
            score = f"dishes: {soups[t]} · points: {pts[t]}" if is_new(layout) else f"soups served: {soups[t]}"
            d.text((x, y + 34), score, font=body_font, fill=(37, 99, 235))
            if scene.get("hats"):
                d.text((x + 200, y + 36), scene["hats"], font=small, fill=(90, 90, 90))
            canvas.paste(frames[t], (x, y + head))
        d.text((pad, H - foot - 6), f"step {t} of {n - 1}  ·  {scene['note']}", font=small, fill=(90, 90, 90))
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
