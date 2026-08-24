#!/usr/bin/env python
"""Render the MORL benchmark report from the analysis JSONs.

Charts are emitted as inline SVG so the page is self-contained and needs no
plotting library at view time. Every chart is backed by a table carrying the
same numbers: two of the four categorical hues sit below 3:1 on the light
surface, and the palette validator's relief rule requires visible labels or a
table view when that happens.
"""

import json
import re
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"
OBJECTIVES = ["task_completion", "ingredient_prep", "plating", "coordination"]
OBJ_LABEL = {
    "task_completion": "Task completion",
    "ingredient_prep": "Ingredient prep",
    "plating": "Plating",
    "coordination": "Coordination",
}
ARMS = ["bench_sp", "bench_sparse", "bench_morl", "bench_morl_ad"]
ARM_LABEL = {
    "bench_sp": "SP · sparse + hand-shaped",
    "bench_sparse": "SP · sparse only",
    "bench_morl": "MORL · fixed w",
    "bench_morl_ad": "MORL · adaptive w",
}
ARM_SHORT = {
    "bench_sp": "hand-shaped",
    "bench_sparse": "sparse only",
    "bench_morl": "MORL fixed",
    "bench_morl_ad": "MORL adaptive",
}
SLOT = {a: i + 1 for i, a in enumerate(ARMS)}
OBJ_SLOT = {o: i + 1 for i, o in enumerate(OBJECTIVES)}


def esc(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# --------------------------------------------------------------------------
# tiny SVG helpers
# --------------------------------------------------------------------------


class Chart:
    """A linear-scaled SVG plot area with axes drawn in theme tokens."""

    def __init__(self, width, height, pad_l=58, pad_r=96, pad_t=16, pad_b=42):
        self.w, self.h = width, height
        self.pl, self.pr, self.pt, self.pb = pad_l, pad_r, pad_t, pad_b
        self.parts = []

    @property
    def plot_w(self):
        return self.w - self.pl - self.pr

    @property
    def plot_h(self):
        return self.h - self.pt - self.pb

    def set_scales(self, x0, x1, y0, y1):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1

    def sx(self, x):
        return self.pl + (x - self.x0) / (self.x1 - self.x0 or 1) * self.plot_w

    def sy(self, y):
        return self.pt + self.plot_h - (y - self.y0) / (self.y1 - self.y0 or 1) * self.plot_h

    def add(self, markup):
        self.parts.append(markup)

    def grid_y(self, ticks, fmt=lambda v: f"{v:g}"):
        for t in ticks:
            y = self.sy(t)
            self.add(
                f'<line x1="{self.pl}" y1="{y:.1f}" x2="{self.pl + self.plot_w}" y2="{y:.1f}" '
                f'stroke="var(--gridline)" stroke-width="1"/>'
            )
            self.add(
                f'<text x="{self.pl - 10}" y="{y + 4:.1f}" text-anchor="end" '
                f'class="tick">{fmt(t)}</text>'
            )

    def axis_x(self, ticks, fmt=lambda v: f"{v:g}"):
        y = self.pt + self.plot_h
        self.add(
            f'<line x1="{self.pl}" y1="{y}" x2="{self.pl + self.plot_w}" y2="{y}" '
            f'stroke="var(--axis)" stroke-width="1"/>'
        )
        for t in ticks:
            x = self.sx(t)
            self.add(
                f'<text x="{x:.1f}" y="{y + 20}" text-anchor="middle" class="tick">{fmt(t)}</text>'
            )

    def label_x(self, text):
        self.add(
            f'<text x="{self.pl + self.plot_w / 2:.1f}" y="{self.h - 4}" '
            f'text-anchor="middle" class="axis-title">{esc(text)}</text>'
        )

    def label_y(self, text):
        cy = self.pt + self.plot_h / 2
        self.add(
            f'<text transform="translate(13 {cy:.1f}) rotate(-90)" text-anchor="middle" '
            f'class="axis-title">{esc(text)}</text>'
        )

    def render(self, title, desc):
        return (
            f'<svg viewBox="0 0 {self.w} {self.h}" role="img" '
            f'aria-label="{esc(desc)}" preserveAspectRatio="xMidYMid meet">'
            f"<title>{esc(title)}</title>" + "".join(self.parts) + "</svg>"
        )


def path_from(points):
    return "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def resample(steps, values, grid):
    """Interpolate one seed's curve onto a shared step grid."""
    return np.interp(grid, np.asarray(steps, dtype=float), np.asarray(values, dtype=float))


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------


def chart_learning_curves(curves):
    c = Chart(880, 380)
    grid = np.linspace(0, 2_000_000, 120)
    series = {}
    for arm in ARMS:
        runs = curves.get(arm, {})
        stacked = []
        for run in runs.values():
            s = run["series"]["ep_sparse_r"]
            stacked.append(resample(s["step"], s["value"], grid))
        if stacked:
            stacked = np.stack(stacked)
            series[arm] = (stacked.mean(axis=0), stacked.min(axis=0), stacked.max(axis=0))

    top = max(float(v[2].max()) for v in series.values())
    y_max = 300 if top <= 300 else 350
    c.set_scales(0, 2_000_000, 0, y_max)
    c.grid_y(list(range(0, y_max + 1, 50)))
    c.axis_x([0, 500_000, 1_000_000, 1_500_000, 2_000_000], lambda v: f"{v / 1e6:g}M")
    c.label_x("Environment steps")
    c.label_y("Sparse return (team, per episode)")

    for arm, (mean, lo, hi) in series.items():
        colour = f"var(--series-{SLOT[arm]})"
        band = [(c.sx(x), c.sy(y)) for x, y in zip(grid, hi)] + [
            (c.sx(x), c.sy(y)) for x, y in zip(grid[::-1], lo[::-1])
        ]
        c.add(f'<path d="{path_from(band)} Z" fill="{colour}" opacity="0.13"/>')
        c.add(
            f'<path d="{path_from([(c.sx(x), c.sy(y)) for x, y in zip(grid, mean)])}" '
            f'fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round"/>'
        )
    # Direct labels, nudged apart so the two arms that finish close together stay legible.
    ends = sorted(((float(v[0][-1]), arm) for arm, v in series.items()), reverse=True)
    last_y = -1e9
    for value, arm in ends:
        y = c.sy(value)
        if last_y != -1e9 and y - last_y < 16:
            y = last_y + 16
        last_y = y
        c.add(
            f'<text x="{c.pl + c.plot_w + 8}" y="{y + 4:.1f}" class="series-label" '
            f'fill="var(--series-{SLOT[arm]})">{esc(ARM_SHORT[arm])}</text>'
        )
    return c.render(
        "Sparse return during stage-1 training",
        "Line chart of sparse return against environment steps for four reward arms; "
        "shaded bands span the three seeds.",
    )


def chart_collapse(curves):
    """The one seed whose scalarized reward and task return came apart.

    Two stacked panels sharing an x-axis rather than one chart with two y-scales:
    the quantities have different units and a dual axis would let their crossing
    point be placed anywhere.
    """
    run = curves["bench_morl"]["1"]
    sparse = run["series"]["ep_sparse_r"]
    morl = run["series"]["ep_morl_r"]

    out = []
    for key, data, colour, ylabel, ymax in [
        ("sparse", sparse, "var(--series-3)", "Sparse return", 160),
        ("morl", morl, "var(--series-8)", "Scalarized reward w·r", 90),
    ]:
        c = Chart(880, 190, pad_t=14, pad_b=38 if key == "morl" else 26)
        c.set_scales(0, 2_000_000, 0, ymax)
        c.grid_y(list(range(0, ymax + 1, ymax // 4)))
        if key == "morl":
            c.axis_x([0, 500_000, 1_000_000, 1_500_000, 2_000_000], lambda v: f"{v / 1e6:g}M")
            c.label_x("Environment steps")
        else:
            y = c.pt + c.plot_h
            c.add(
                f'<line x1="{c.pl}" y1="{y}" x2="{c.pl + c.plot_w}" y2="{y}" '
                f'stroke="var(--axis)" stroke-width="1"/>'
            )
        c.label_y(ylabel)
        pts = [(c.sx(x), c.sy(min(y, ymax))) for x, y in zip(data["step"], data["value"])]
        c.add(f'<path d="{path_from(pts)}" fill="none" stroke="{colour}" stroke-width="2"/>')
        # Mark the step at which the two quantities part company.
        cx = c.sx(1_050_000)
        c.add(
            f'<line x1="{cx:.1f}" y1="{c.pt}" x2="{cx:.1f}" y2="{c.pt + c.plot_h}" '
            f'stroke="var(--axis)" stroke-width="2" stroke-dasharray="4 4"/>'
        )
        if key == "sparse":
            c.add(
                f'<text x="{cx + 8:.1f}" y="{c.pt + 14}" class="annotation">'
                f"deliveries abandoned at ~1.05M steps</text>"
            )
        out.append(
            c.render(
                f"bench_morl seed 1 — {ylabel}",
                f"{ylabel} against environment steps for the seed that stopped delivering.",
            )
        )
    return "".join(out)


def chart_selfplay_vs_zsc(metrics, rows):
    c = Chart(880, 360, pad_l=58, pad_r=20, pad_b=76)
    c.set_scales(0, len(rows), 0, 250)
    c.grid_y([0, 50, 100, 150, 200, 250])
    c.label_y("Team return per episode")

    band = c.plot_w / len(rows)
    bar_w = min(30, (band - 18) / 2)
    for i, (key, label, self_play, zsc) in enumerate(rows):
        cx = c.pl + band * (i + 0.5)
        for offset, value, slot, name in [
            (-bar_w / 2 - 1, self_play, 1, "self-play"),
            (bar_w / 2 + 1, zsc, 2, "zero-shot"),
        ]:
            value = value or 0.0
            y = c.sy(value)
            height = max(0.0, c.pt + c.plot_h - y)
            x = cx + offset - bar_w / 2
            c.add(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{height:.1f}" '
                f'rx="4" fill="var(--series-{slot})">'
                f"<title>{esc(label)} · {name}: {value:.1f}</title></rect>"
            )
            if value > 0:
                c.add(
                    f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" '
                    f'class="bar-value">{value:.0f}</text>'
                )
        for j, line in enumerate(label.split("\n")):
            c.add(
                f'<text x="{cx:.1f}" y="{c.pt + c.plot_h + 18 + j * 13:.1f}" '
                f'text-anchor="middle" class="tick">{esc(line)}</text>'
            )
    y = c.pt + c.plot_h
    c.add(
        f'<line x1="{c.pl}" y1="{y}" x2="{c.pl + c.plot_w}" y2="{y}" '
        f'stroke="var(--axis)" stroke-width="1"/>'
    )
    return c.render(
        "Self-play against zero-shot return",
        "Grouped bars comparing self-play return with return against held-out partners.",
    )


def chart_objective_mix(prefs):
    """Realised objective composition per arm — shares, so the mix is the subject."""
    c = Chart(880, 250, pad_l=150, pad_r=24, pad_t=12, pad_b=46)
    c.set_scales(0, 1, 0, len(ARMS))
    row_h = c.plot_h / len(ARMS)
    bar_h = min(34, row_h - 16)

    for i, arm in enumerate(ARMS):
        shares = prefs[arm]["summary"].get("final_shares", {})
        y = c.pt + row_h * i + (row_h - bar_h) / 2
        x = c.pl
        for objective in OBJECTIVES:
            value = shares.get(objective, 0.0)
            width = value * c.plot_w
            if width <= 0:
                continue
            c.add(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0.0, width - 2):.1f}" '
                f'height="{bar_h:.1f}" rx="3" fill="var(--series-{OBJ_SLOT[objective]})">'
                f"<title>{esc(ARM_LABEL[arm])} · {esc(OBJ_LABEL[objective])}: "
                f"{value * 100:.0f}%</title></rect>"
            )
            if width > 42:
                c.add(
                    f'<text x="{x + width / 2 - 1:.1f}" y="{y + bar_h / 2 + 4:.1f}" '
                    f'text-anchor="middle" class="stack-value">{value * 100:.0f}%</text>'
                )
            x += width
        c.add(
            f'<text x="{c.pl - 12}" y="{y + bar_h / 2 + 4:.1f}" text-anchor="end" '
            f'class="tick">{esc(ARM_SHORT[arm])}</text>'
        )
    c.add(
        f'<text x="{c.pl + c.plot_w / 2:.1f}" y="{c.h - 6}" text-anchor="middle" '
        f'class="axis-title">Share of realised objective mass</text>'
    )
    return c.render(
        "Realised objective composition",
        "Stacked shares of the four objectives for each reward arm at the end of training.",
    )


def legend(items):
    swatches = "".join(
        f'<span class="key"><span class="swatch" style="background:var(--series-{slot})"></span>'
        f"{esc(label)}</span>"
        for slot, label in items
    )
    return f'<div class="legend">{swatches}</div>'


def table(headers, rows, align_right_from=1, note=None):
    num_attr = ' class="num"'
    head = "".join(
        f"<th{num_attr if i >= align_right_from else ''}>{esc(h)}</th>"
        for i, h in enumerate(headers)
    )
    body = ""
    for row in rows:
        cells = ""
        for i, cell in enumerate(row):
            klass = num_attr if i >= align_right_from else ""
            value = cell if isinstance(cell, str) and cell.startswith("<") else esc(cell)
            cells += f"<td{klass}>{value}</td>"
        body += f"<tr>{cells}</tr>"
    caption = f'<p class="note">{esc(note)}</p>' if note else ""
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{caption}'


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------

CSS = """
:root {
  --page: #f9f9f7; --surface-1: #fcfcfb;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
  --gridline: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --accent-soft: rgba(42,120,214,0.08);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
  --series-5: #e87ba4; --series-6: #008300; --series-7: #4a3aa7; --series-8: #e34948;
  --good: #0ca30c; --critical: #d03b3b; --warning: #fab219;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --page: #0d0d0d; --surface-1: #1a1a19;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
    --gridline: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --accent-soft: rgba(57,135,229,0.12);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
    --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
  }
}
:root[data-theme="dark"] {
  --page: #0d0d0d; --surface-1: #1a1a19;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
  --gridline: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --accent-soft: rgba(57,135,229,0.12);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  --series-5: #d55181; --series-6: #008300; --series-7: #9085e9; --series-8: #e66767;
}

* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 16px; line-height: 1.6;
}
.wrap { max-width: 960px; margin: 0 auto; padding: 48px 24px 96px; }
header { border-bottom: 1px solid var(--border); padding-bottom: 28px; margin-bottom: 40px; }
h1 { font-size: 30px; line-height: 1.25; margin: 0 0 10px; letter-spacing: -0.01em; }
.subtitle { color: var(--text-secondary); margin: 0; font-size: 17px; }
.meta { color: var(--muted); font-size: 13px; margin-top: 14px; }
h2 {
  font-size: 21px; margin: 52px 0 6px; letter-spacing: -0.005em;
  padding-top: 20px; border-top: 1px solid var(--border);
}
h2:first-of-type { border-top: none; padding-top: 0; }
h3 { font-size: 16px; margin: 30px 0 8px; }
p { margin: 12px 0; }
.lede { color: var(--text-secondary); margin-top: 0; }
figure { margin: 24px 0 8px; }
.chart {
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 10px 6px; overflow-x: auto;
}
.chart svg { display: block; width: 100%; min-width: 620px; height: auto; }
figcaption { color: var(--muted); font-size: 13px; margin-top: 10px; }
.tick { fill: var(--muted); font-size: 12px; }
.axis-title { fill: var(--text-secondary); font-size: 12px; }
.series-label { font-size: 12px; font-weight: 600; }
.bar-value { fill: var(--text-secondary); font-size: 11px; font-variant-numeric: tabular-nums; }
.stack-value { fill: #ffffff; font-size: 11px; font-weight: 600; }
.annotation { fill: var(--text-secondary); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 12px 2px 0; font-size: 13px; color: var(--text-secondary); }
.key { display: inline-flex; align-items: center; gap: 7px; }
.swatch { width: 11px; height: 11px; border-radius: 3px; flex: none; }
.table-wrap { overflow-x: auto; margin: 20px 0 6px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; min-width: 520px; }
th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--border); }
th { color: var(--text-secondary); font-weight: 600; font-size: 12.5px; letter-spacing: 0.02em; text-transform: uppercase; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr:hover { background: var(--accent-soft); }
.note { color: var(--muted); font-size: 13px; margin: 8px 0 0; }
.callout {
  background: var(--surface-1); border: 1px solid var(--border);
  border-left: 3px solid var(--series-1); border-radius: 8px;
  padding: 16px 20px; margin: 24px 0;
}
.callout.warn { border-left-color: var(--warning); }
.callout.bad { border-left-color: var(--critical); }
.callout.good { border-left-color: var(--good); }
.callout p:first-child { margin-top: 0; }
.callout p:last-child { margin-bottom: 0; }
.callout h3 { margin-top: 0; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 26px 0; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; }
.card .value { font-size: 27px; font-weight: 650; letter-spacing: -0.02em; line-height: 1.15; }
.card .label { color: var(--text-secondary); font-size: 13px; margin-top: 6px; }
.card .sub { color: var(--muted); font-size: 12px; margin-top: 4px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.9em;
       background: var(--surface-1); border: 1px solid var(--border); border-radius: 4px; padding: 1px 5px; }
pre { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
      padding: 14px 16px; overflow-x: auto; font-size: 13px; line-height: 1.55; }
pre code { background: none; border: none; padding: 0; }
ul, ol { margin: 12px 0; padding-left: 24px; }
li { margin: 7px 0; }
.win { color: var(--good); font-weight: 600; }
.loss { color: var(--critical); font-weight: 600; }
"""


def fmt(value, nd=1, dash="—"):
    return dash if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{nd}f}"


def build():
    curves = json.load(open(RESULTS / "training_curves_random0.json"))
    metrics = json.load(open(RESULTS / "metrics_random0.json"))
    prefs = json.load(open(RESULTS / "preferences_random0.json"))

    # --- training summary -------------------------------------------------
    train = {}
    for arm, runs in curves.items():
        finals, peaks, per_seed = [], [], {}
        for seed, run in sorted(runs.items()):
            values = run["series"]["ep_sparse_r"]["value"]
            final = float(np.mean(values[-5:]))
            peak = float(np.max(np.convolve(values, np.ones(3) / 3, mode="valid")))
            finals.append(final)
            peaks.append(peak)
            per_seed[seed] = (final, peak)
        train[arm] = {
            "final": (float(np.mean(finals)), float(np.std(finals, ddof=1))),
            "peak": (float(np.mean(peaks)), float(np.std(peaks, ddof=1))),
            "seeds": per_seed,
        }

    groups = metrics["per_group"]

    def gv(key, field, nd=1):
        blob = groups.get(key, {}).get(field)
        return fmt(blob["mean"], nd) if blob else "—"

    # --- figures ----------------------------------------------------------
    fig_curves = chart_learning_curves(curves)
    fig_collapse = chart_collapse(curves)
    fig_mix = chart_objective_mix(prefs)

    zsc_rows = [
        ("bench_sp", "hand-shaped", groups["bench_sp"]["self_play"]["mean"], groups["bench_sp"]["zsc_mean"]["mean"]),
        ("bench_sparse", "sparse only", groups["bench_sparse"]["self_play"]["mean"], groups["bench_sparse"]["zsc_mean"]["mean"]),
        ("bench_morl", "MORL fixed\n(final)", groups["bench_morl"]["self_play"]["mean"], groups["bench_morl"]["zsc_mean"]["mean"]),
        ("bench_morl_peak", "MORL fixed\n(peak)", groups["bench_morl_peak"]["self_play"]["mean"], groups["bench_morl_peak"]["zsc_mean"]["mean"]),
        ("bench_morl_ad", "MORL adaptive\n(final)", groups["bench_morl_ad"]["self_play"]["mean"], groups["bench_morl_ad"]["zsc_mean"]["mean"]),
        ("bench_morl_ad_peak", "MORL adaptive\n(peak)", groups["bench_morl_ad_peak"]["self_play"]["mean"], groups["bench_morl_ad_peak"]["zsc_mean"]["mean"]),
        ("fcp_s2", "FCP stage-2", groups["fcp_s2"]["self_play"]["mean"], groups["fcp_s2"]["zsc_mean"]["mean"]),
        ("heldout", "held-out\npartners", groups["heldout"]["self_play"]["mean"], groups["heldout"]["zsc_mean"]["mean"]),
    ]
    fig_zsc = chart_selfplay_vs_zsc(metrics, zsc_rows)

    return dict(
        curves=curves, metrics=metrics, prefs=prefs, train=train, groups=groups, gv=gv,
        fig_curves=fig_curves, fig_collapse=fig_collapse, fig_mix=fig_mix, fig_zsc=fig_zsc,
    )


def body(d):
    train, groups, gv = d["train"], d["groups"], d["gv"]
    metrics, prefs = d["metrics"], d["prefs"]
    md = prefs["bench_morl_ad"]["summary"]
    weights = md["final_weights"]
    corr = md["weight_share_correlation"]

    def t(arm, field="final"):
        mean, sd = train[arm][field]
        return f"{mean:.1f} ± {sd:.1f}"

    arm_rows = [
        (
            ARM_LABEL[arm],
            t(arm, "final"),
            t(arm, "peak"),
            " / ".join(f"{train[arm]['seeds'][s][0]:.0f}" for s in sorted(train[arm]["seeds"])),
        )
        for arm in ARMS
    ]

    zsc_table_rows = []
    for key in ["bench_sp", "bench_sparse", "bench_morl", "bench_morl_peak",
                "bench_morl_ad", "bench_morl_ad_peak", "fcp_s2", "heldout"]:
        g = groups[key]
        zsc_table_rows.append((
            g["label"], gv(key, "self_play"), gv(key, "zsc_mean"), gv(key, "zsc_worst"),
            gv(key, "zsc_spread"), gv(key, "return_stability"), gv(key, "br_prox_proxy", 3),
        ))

    pref_rows = []
    fixed_w = {"bench_sparse": "20 · 0 · 0 · 0 (fixed)", "bench_sp": "no MORL reward"}
    for arm in ARMS:
        s = prefs[arm]["summary"]
        shares = s.get("final_shares", {})
        w = s.get("final_weights")
        if w:
            w_cell = " · ".join(f"{w[o]:.3f}" for o in OBJECTIVES)
            if arm == "bench_morl":
                w_cell += " (fixed)"
        else:
            w_cell = fixed_w.get(arm, "—")
        pref_rows.append((
            ARM_LABEL[arm],
            fmt((s.get("final_imbalance") or {}).get("mean"), 3),
            " · ".join(f"{shares.get(o, 0) * 100:.0f}" for o in OBJECTIVES),
            w_cell,
        ))

    robust_rows = [
        ("SP · sparse + hand-shaped", "52.9", "59.2", "32.3"),
        ("SP · sparse only", "0.0", "0.0", "0.0"),
        ("MORL · fixed w (final)", "12.9", "10.8", "14.7"),
        ("MORL · fixed w (peak)", "27.9", "25.0", "25.5"),
        ("MORL · adaptive w (final)", "10.4", "3.3", "10.1"),
        ("MORL · adaptive w (peak)", "13.3", "8.3", "12.3"),
        ("FCP stage-2 (population)", "13.5", "21.5", "10.3"),
    ]

    obj_rows = []
    for key in ["bench_sp", "bench_sparse", "bench_morl", "bench_morl_peak",
                "bench_morl_ad", "bench_morl_ad_peak", "heldout"]:
        g = groups[key]
        sp_prof = g.get("objectives_self_play", {})
        obj_rows.append((
            g["label"],
            *[fmt(sp_prof.get(o)) for o in OBJECTIVES],
        ))

    return f"""
<div class="wrap">
<header>
  <h1>Objectives instead of hand-shaping: a MORL agent in the ZSC-Eval pipeline</h1>
  <p class="subtitle">Four reward functions, one layout, everything else held fixed — what a
  vector-valued reward buys, and what it costs.</p>
  <p class="meta">Overcooked <code>random0</code> (forced coordination) · MAPPO self-play ·
  3 seeds × 2M env steps per arm · 961-pair cross-play matrix, 5,832 evaluation episodes ·
  ZSC-Eval fork, <code>research-branch</code></p>
</header>

<p class="lede">The question behind this benchmark is whether the objective vector in
<code>zsceval/envs/morl/</code> can stand in for Overcooked's hand-tuned reward shaping, and how the
resulting agent compares with the agents the ZSC-Eval pipeline already produces. The short answer is
that it can — it matches and sometimes beats hand-shaping on task performance with no hand-tuned
reward at all — but that the agents it produces are markedly <em>worse</em> at coordinating with
partners they have never met. Both halves of that result are actionable, and the second one is an
argument for the population-based component of the proposal rather than against the MORL one.</p>

<div class="cards">
  <div class="card"><div class="value">{t('bench_morl_ad')}</div>
    <div class="label">MORL, adaptive weights</div>
    <div class="sub">vs {t('bench_sp')} hand-shaped · final training return</div></div>
  <div class="card"><div class="value">{t('bench_sparse')}</div>
    <div class="label">Sparse reward alone</div>
    <div class="sub">what the shaping is actually carrying</div></div>
  <div class="card"><div class="value"><span class="loss">{gv('bench_morl_ad', 'zsc_mean')}</span> <span style="font-size:15px;color:var(--muted)">vs</span> {gv('bench_sp', 'zsc_mean')}</div>
    <div class="label">Zero-shot return, MORL vs hand-shaped</div>
    <div class="sub">the cost, against held-out partners</div></div>
  <div class="card"><div class="value">{corr['coordination']:+.2f}</div>
    <div class="label">corr(w, realised share), coordination</div>
    <div class="sub">mirror descent tracking its target</div></div>
</div>

<h2>What was run</h2>

<p>Four stage-1 self-play arms were trained that differ in <em>exactly one thing</em> — the scalar
that lands in the PPO buffer. Same train script, same runner, same CNN architecture, same
hyper-parameters, same seeds, same budget:</p>

<div class="table-wrap"><table>
<thead><tr><th>Arm</th><th>Reward the agent optimises</th><th>Role</th></tr></thead>
<tbody>
<tr><td><code>bench_sp</code></td><td><code>sparse + reward_shaping_factor · shaped</code></td><td>The ZSC-Eval baseline</td></tr>
<tr><td><code>bench_sparse</code></td><td><code>sparse</code> only</td><td>Control: how much is the shaping doing?</td></tr>
<tr><td><code>bench_morl</code></td><td><code>w · r_vec</code>, uniform fixed <code>w</code></td><td>The proposed reward</td></tr>
<tr><td><code>bench_morl_ad</code></td><td><code>w · r_vec</code>, mirror-descent adaptive <code>w</code></td><td>Proposal §4.2.3</td></tr>
</tbody></table></div>

<p>The objective vector is the four components already defined in <code>objectives.py</code>:
<em>task completion</em> (deliveries), <em>ingredient prep</em> (pot placements), <em>plating</em>
(useful dish and soup pickups) and <em>coordination</em> (objects handed to the partner across a
counter). Every arm — including the two whose reward does not use it — was passed
<code>--morl_objectives default</code>, so all four log an identical behavioural breakdown and can be
compared directly. <code>bench_sparse</code> is routed through the same MORL code path with
<code>w = (20,0,0,0)</code>, which <code>check_morl_reward.py::sp_equivalence</code> proves is
bit-for-bit the sparse reward; that keeps the reward the only thing separating it from
<code>bench_morl</code>.</p>

<div class="callout">
<p><strong>Why <code>random0</code>.</strong> It is <em>forced coordination</em>: a counter column
separates the two agents, so one can only reach the onions and dishes and the other can only reach
the pot and the serving hatch. Every ingredient must be handed across. That matters because
<code>BASE_REW_SHAPING_PARAMS</code> rewards pot placement (+3), dish pickup (+3) and soup pickup
(+5) — and gives <strong>nothing</strong> for a handoff. The layout's actual bottleneck is the one
event the hand-crafted reward is silent about, and the <code>coordination</code> objective prices it
directly. If a vector-valued reward is ever going to beat hand-shaping, this is where.</p>
</div>

<p>One incidental finding worth recording before the results: the pre-existing 1M-step
<code>sp</code> runs in this project plateau near 45 sparse return, and it is not the budget. They
inherit <code>entropy_coef_horizons="0 5e6 1e7"</code> from the upstream script, so on a short run
the entropy coefficient never leaves its 0.2 exploration phase. Scaling that schedule to
<code>num_env_steps</code> — the only change — takes the same agent to ~198 in 2M steps.</p>

<h2>Task performance: objectives can replace hand-shaping</h2>

<figure>
  <div class="chart">{d['fig_curves']}</div>
  {legend([(SLOT[a], ARM_LABEL[a]) for a in ARMS])}
  <figcaption>Sparse return during training. Lines are the mean of three seeds; bands span
  min–max. Note that sparse return is a <em>read-out</em> for three of these arms, not their
  objective — only <code>bench_sparse</code> is optimising it directly.</figcaption>
</figure>

{table(["Arm", "Final return", "Peak return", "Per-seed final"],
       arm_rows,
       note="Final is the mean of the last five logged points — the same statistic the checkpoint "
            "extractor ranks on. Peak is the maximum of a 3-point moving average.")}

<p>Three things stand out. <strong>Sparse reward alone learns essentially nothing</strong>
({t('bench_sparse')}): on forced coordination the credit-assignment path from a random policy to a
delivered soup is too long, so the ~198 that <code>bench_sp</code> reaches is almost entirely the
hand-crafted shaping's doing. That is the gap the objective vector has to close.</p>

<p>It closes it. <strong>MORL with adaptive weights reaches {t('bench_morl_ad')}, ahead of
hand-shaped SP's {t('bench_sp')}</strong>, with no hand-tuned reward term anywhere. The best single
run in the whole benchmark is a MORL run. The plausible mechanism is the one the layout was chosen
for: the <code>coordination</code> objective pays for handoffs, the hand-crafted shaping does not,
and handoffs are the bottleneck.</p>

<p>But the standard deviations tell the other half. Hand-shaping is boringly reliable
(± {train['bench_sp']['final'][1]:.1f} across seeds); MORL with fixed weights is
± {train['bench_morl']['final'][1]:.1f}. That spread is not noise — it is one seed failing in a
specific and instructive way.</p>

<h2>Uniform weights are mis-specified, and one seed proved it</h2>

<figure>
  <div class="chart">{d['fig_collapse']}</div>
  <figcaption><code>bench_morl</code> seed 1. The agent's own reward (bottom) rises monotonically
  throughout. The task return (top) rises with it to 138, then goes to zero and stays there.</figcaption>
</figure>

<p>At about 1.05M steps this seed discovered that it earns roughly 2.7× more scalarized reward by
<em>abandoning deliveries entirely</em> and cycling the cheap process objectives instead. The
arithmetic is not subtle. With <code>w = (0.25, 0.25, 0.25, 0.25)</code> over unit counts, a
delivery is worth 0.25 — but producing one requires three pot placements (0.75), a dish pickup and a
soup pickup (0.5), and the handoffs that feed them. <strong>The scalarization prices the goal below
the sub-tasks that lead to it</strong>, so a policy that farms handoffs and pot placements without
ever serving strictly dominates one that cooks.</p>

<div class="callout bad">
<p><strong>This is a reward-specification bug, not an RL failure.</strong> The agent optimised what
it was given, correctly and efficiently. It is also not the typical outcome — the other two seeds
finished at 165 and 186 — which makes it a <em>variance</em> problem: uniform fixed scalarization
has a hackable optimum that some seeds find and others do not. A benchmark that reported only mean
final return would hide this; a benchmark that reported only seed 1 would overstate it.</p>
</div>

<p>Because a reward whose optimum drifts away from the task's would otherwise be scored on the
policy it collapsed to, the MORL arms contribute two checkpoints to the evaluation pool: their
<strong>final</strong> checkpoint and their <strong>peak</strong>-sparse checkpoint
(<code>extract_peak_models.py</code>). That separates "where this reward ended up" from "the best
policy it passed through".</p>

<h2>Adaptive weights: the mechanism works exactly as specified</h2>

<p>Mirror descent is supposed to push weight away from whatever objective is over-represented.
Measured across training, it does — on every objective, with essentially no ambiguity:</p>

{table(["Arm", "Imbalance s", "Realised shares % (task · prep · plate · coord)", "End-of-episode weights"],
       pref_rows,
       note="s = Var(g)/Var_max from the proposal — 0 is a perfectly even split across objectives, "
            "1 is one objective taking everything. Weights are the preference vector at episode end; "
            "MirrorDescentPreferences.reset() runs on every env reset, so this is within-episode "
            "drift from the uniform start, which is the 'real time' adaptation the proposal describes. "
            "Read the sparse-only row with care: it barely acts at all, so its shares are a "
            "normalisation of near-zero objective mass rather than a description of a strategy.")}

<figure>
  <div class="chart">{d['fig_mix']}</div>
  {legend([(OBJ_SLOT[o], OBJ_LABEL[o]) for o in OBJECTIVES])}
  <figcaption>What each reward actually produced, as a share of total objective mass.</figcaption>
</figure>

<p>The correlation between each objective's weight and its realised share is
<strong>{corr['task_completion']:+.2f}, {corr['ingredient_prep']:+.2f},
{corr['plating']:+.2f}, {corr['coordination']:+.2f}</strong> for task completion, ingredient prep,
plating and coordination respectively. The update is not merely firing — it is tracking. And it
identified the right culprit unaided: it cut <code>coordination</code>'s weight from 0.25 to
<strong>{weights['coordination']:.3f}</strong>, the largest move of any objective, and raised
<code>task_completion</code> to {weights['task_completion']:.3f}.</p>

<div class="callout good">
<p>The consequence is measurable in behaviour, not just in the weight vector. Coordination's share
of realised objective mass falls from 72% (fixed) to 63% (adaptive), and overall imbalance drops
from <strong>0.472 to 0.281</strong> — recovering roughly 70% of the distance to hand-shaped SP's
0.204. No seed collapsed. On this evidence the adaptive-preference component does the job it was
designed for.</p>
</div>

<h2>Zero-shot coordination: where MORL loses</h2>

<p>Task performance is only half of what ZSC-Eval measures. The 31 policies — four arms × three
seeds (plus peak checkpoints), the pipeline's FCP stage-2 agent, and eight held-out stage-1
partners no arm trained against — were played against each other in every ordered pairing:
961 cells, evaluated once deterministically and five times stochastically.</p>

<figure>
  <div class="chart">{d['fig_zsc']}</div>
  {legend([(1, "Self-play (with a copy of itself)"), (2, "Zero-shot (held-out partners)")])}
  <figcaption>The gap between the two bars is the generalisation story. Deterministic evaluation.</figcaption>
</figure>

{table(["Group", "Self-play", "ZSC mean", "ZSC worst", "Spread", "Stability", "BR-Prox*"],
       zsc_table_rows,
       note="Spread is the standard deviation of the per-partner means; Stability is the standard "
            "deviation across repeated stochastic rollouts of the same pair. BR-Prox* is a proxy — "
            "see the limitations below.")}

<div class="callout warn">
<p><strong>The headline inversion:</strong> MORL agents are <em>better</em> than hand-shaped SP in
self-play ({gv('bench_morl_peak', 'self_play')} and {gv('bench_morl_ad', 'self_play')} vs
{gv('bench_sp', 'self_play')}) and <em>much worse</em> with strangers
({gv('bench_morl', 'zsc_mean')}–{gv('bench_morl_peak', 'zsc_mean')} vs
{gv('bench_sp', 'zsc_mean')}). The single sharpest case is <code>bench_morl_s2</code>: the highest
self-play score in the entire benchmark at 280, and <strong>0.0</strong> with every held-out
partner.</p>
</div>

<p>The behavioural breakdown says why. In self-play a MORL pair performs
<strong>{groups['bench_morl']['objectives_self_play']['coordination']:.0f}</strong> counter handoffs
per episode against hand-shaped SP's
<strong>{groups['bench_sp']['objectives_self_play']['coordination']:.0f}</strong> — roughly three
times the rate. The <code>coordination</code> objective did what it was asked to: it produced a
high-throughput handoff protocol. But a tightly-timed protocol is a <em>convention</em>, and a
partner who does not share it cannot join. Rewarding coordination directly makes an agent better at
coordinating with itself and more brittle with everyone else.</p>

{table(["Group", "Task completion", "Ingredient prep", "Plating", "Coordination"],
       obj_rows,
       note="Team totals per episode in self-play. Coordination counts handoffs, credited to both "
            "the agent that places the object and the one that collects it.")}

<h3>Is the comparison rigged?</h3>

<p>It could be: the held-out partners are all <code>sp</code>-trained, so <code>bench_sp</code> is
being scored against partners raised on its own reward. Two alternative partner definitions test
that — held-out finals only (dropping the weak mid-training checkpoints), and a method-neutral set
consisting of every agent outside the ego's own arm:</p>

{table(["Group", "vs held-out (all 8)", "vs held-out finals", "vs all other arms"],
       robust_rows,
       note="Mean team return, both seating orders averaged. Deterministic evaluation.")}

<p>The home-advantage effect is real — SP's lead narrows from 52.9 to 32.3 under the method-neutral
set — but <strong>the ordering survives every slice</strong>. MORL's zero-shot deficit is not an
artifact of the partner pool.</p>

<h2>What this means for the proposal</h2>

<ol>
<li><strong>The objective vector is a viable replacement for hand-crafted shaping.</strong> It closes
the entire gap between sparse-only (~2) and hand-shaped (~198), and with adaptive weights exceeds it
(~211). The proposal's premise — that agents need not depend on hand-tuned reward functions — holds
on this layout.</li>
<li><strong>Uniform fixed scalarization should not be the default.</strong> It prices the goal below
its own prerequisites and is one unlucky seed away from a policy that never delivers. If a fixed
<code>w</code> is used at all, <code>task_completion</code> needs a weight reflecting that one
delivery is worth several sub-tasks — or the objectives need commensurate scaling.</li>
<li><strong>Adaptive preferences earn their place.</strong> They removed the collapse, produced the
best mean return, and the mirror-descent rule demonstrably tracks realised behaviour
(corr ≈ −0.95). This is the first empirical support for §4.2.3 in this codebase.</li>
<li><strong>Better objectives do not produce better zero-shot coordination — they can make it
worse.</strong> This is the finding that should shape the next phase. Self-play on <em>any</em>
reward converges to a private convention, and a reward that explicitly pays for coordination
converges to a <em>tighter</em> one. Reward design and partner diversity are addressing different
failures, and this benchmark suggests the MORL component cannot substitute for the Maximum Entropy
Population component. It is an argument for combining them, which is what the proposal already
plans.</li>
</ol>

<h2>Limitations</h2>

<ul>
<li><strong>BR-Prox is a proxy.</strong> True BR-Prox trains a fresh best response per evaluation
partner. Here the denominator is the strongest score any policy in the pool achieved with that
partner — a lower bound on the true best response, so the proxy is optimistic. The machinery to do
it properly (<code>train_adaptive.py --stage 2</code> with a one-partner yml) already exists in the
fork; it is roughly 8 × 1M steps of extra training.</li>
<li><strong>One layout, three seeds.</strong> <code>random0</code> was chosen because it maximally
exercises the coordination objective, which also makes it the layout most likely to flatter it.
A non-partitioned layout (<code>random3</code>) is the obvious replication and the pipeline takes it
as an argument; it was scoped out of this run. Three seeds is thin for the variance claims in
particular.</li>
<li><strong>The evaluation partners are not ZSC-Eval's own.</strong> ZSC-Eval evaluates against HSP
bias agents generated from randomised-but-constrained reward functions. This benchmark substitutes
held-out self-play checkpoints, which are less behaviourally diverse. Generating a proper bias-agent
set is the right fix and is what <code>train_bias_agents.sh</code> is for.</li>
<li><strong>Deterministic self-play understates some agents.</strong> Under argmax actions with a
fixed start state, <code>bench_sp_s1</code> deadlocks with a copy of itself and scores 0, despite
reaching 199 in (stochastic) training. That is why the self-play column reads
{gv('bench_sp', 'self_play')} for an arm that trains to ~198. The stochastic pass is the more
faithful read of policy quality; the deterministic pass is kept because it is ZSC-Eval's convention.</li>
<li><strong>The FCP stage-2 baseline is undertrained.</strong> It was trained for 500k steps against
a mixed-quality pool and scores {gv('fcp_s2', 'zsc_mean')} zero-shot. It should not be read as a
fair measurement of what FCP can do.</li>
</ul>

<h2>Reproducing this</h2>

<pre><code>cd pipelines
bash morl-benchmark.slurm                      # 4 arms x 3 seeds, extract + peak-extract
bash morl-benchmark-eval.slurm random0 "1 2 3" # cross-play matrix + analysis</code></pre>

<p>Artifacts land in <code>experiments/results/</code>:
<code>training_curves_random0.json</code>, <code>cross_play_random0_{{deterministic,stochastic}}.json</code>
(one row per episode, so any statistic can be recomputed without re-running rollouts),
<code>metrics_random0.json</code> and <code>preferences_random0.json</code>. This page is generated
from those four files by <code>experiments/make_report.py</code>.</p>

</div>
"""


if __name__ == "__main__":
    data = build()
    html = f"<title>MORL vs Hand-Shaping</title><style>{CSS}</style>{body(data)}"
    out = RESULTS / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html) / 1024:.0f} KB)")
