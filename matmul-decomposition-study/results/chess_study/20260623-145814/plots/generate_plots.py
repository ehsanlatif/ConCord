#!/usr/bin/env python3
"""Generate SVG plots and a short insights report for the chess study run."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parents[1]
PLOT_DIR = Path(__file__).resolve().parent
SUMMARY_JSON = RUN_DIR / "study_summary.json"
SUMMARY_CSV = RUN_DIR / "study_summary.csv"

COLORS = {
    "ink": "#172033",
    "muted": "#64748B",
    "grid": "#E2E8F0",
    "axis": "#94A3B8",
    "panel": "#F8FAFC",
    "success": "#2F855A",
    "fail": "#C2410C",
    "pass": "#2F855A",
    "miss": "#DC2626",
    "ungradeable": "#CBD5E1",
    "tokens": "#2563EB",
    "usd": "#7C3AED",
    "elapsed": "#D97706",
    "calls": "#0F766E",
    "atom": "#EA580C",
    "fidelity": "#0891B2",
    "frontier": "#111827",
    "execution": "#2563EB",
    "verification": "#F59E0B",
    "splitter": "#7C3AED",
    "combiner": "#0F766E",
}


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def fmt_int(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{int(round(value)):,}"


def fmt_k(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1000:.0f}k"


def fmt_usd(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:.2f}"


def fmt_pct(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.0f}%"


def fmt_min(seconds: float | int | None) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds / 60:.1f}m"


def nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10**exponent
    fraction = value / base
    if fraction <= 1:
        nice = 1
    elif fraction <= 2:
        nice = 2
    elif fraction <= 5:
        nice = 5
    else:
        nice = 10
    return nice * base


def svg_root(width: int, height: int, title: str, subtitle: str = "") -> list[str]:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        "<style>",
        "text { font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; fill: #172033; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".subtitle { font-size: 12px; fill: #64748B; }",
        ".label { font-size: 12px; fill: #334155; }",
        ".small { font-size: 10px; fill: #64748B; }",
        ".axis { stroke: #94A3B8; stroke-width: 1; }",
        ".grid { stroke: #E2E8F0; stroke-width: 1; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="36" y="34" class="title">{esc(title)}</text>',
    ]
    if subtitle:
        lines.append(f'<text x="36" y="56" class="subtitle">{esc(subtitle)}</text>')
    return lines


def save_svg(name: str, lines: list[str]) -> Path:
    lines.append("</svg>")
    path = PLOT_DIR / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def text(lines: list[str], x: float, y: float, value: object, size: int = 12, anchor: str = "start",
         weight: int | str = "normal", fill: str | None = None, rotate: float | None = None,
         css_class: str | None = None) -> None:
    style = f"font-size:{size}px;font-weight:{weight};"
    if fill:
        style += f"fill:{fill};"
    attrs = [f'x="{x:.1f}"', f'y="{y:.1f}"', f'text-anchor="{anchor}"', f'style="{style}"']
    if css_class:
        attrs.append(f'class="{css_class}"')
    if rotate is not None:
        attrs.append(f'transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"')
    lines.append(f'<text {" ".join(attrs)}>{esc(value)}</text>')


def line(lines: list[str], x1: float, y1: float, x2: float, y2: float,
         stroke: str, width: float = 1.0, dash: str | None = None) -> None:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    lines.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"{dash_attr}/>'
    )


def rect(lines: list[str], x: float, y: float, w: float, h: float, fill: str,
         stroke: str | None = None, width: float = 1.0, radius: float = 0.0,
         opacity: float | None = None) -> None:
    stroke_attr = f' stroke="{stroke}" stroke-width="{width:.1f}"' if stroke else ""
    radius_attr = f' rx="{radius:.1f}" ry="{radius:.1f}"' if radius else ""
    opacity_attr = f' opacity="{opacity:.2f}"' if opacity is not None else ""
    lines.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}"{stroke_attr}{radius_attr}{opacity_attr}/>'
    )


def circle(lines: list[str], x: float, y: float, r: float, fill: str,
           stroke: str = "white", width: float = 2.0) -> None:
    lines.append(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"/>'
    )


def polyline(lines: list[str], points: list[tuple[float, float]], color: str, width: float = 3.0,
             dash: str | None = None) -> None:
    if len(points) < 2:
        return
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    lines.append(
        f'<polyline points="{point_text}" fill="none" stroke="{color}" '
        f'stroke-width="{width:.1f}" stroke-linecap="round" stroke-linejoin="round"{dash_attr}/>'
    )


def legend(lines: list[str], x: float, y: float, items: list[tuple[str, str]]) -> None:
    cx = x
    for label, color in items:
        rect(lines, cx, y - 9, 12, 12, color, radius=2)
        text(lines, cx + 18, y + 2, label, size=12, fill=COLORS["muted"])
        cx += 18 + len(label) * 7.2 + 22


def plot_area_axes(lines: list[str], left: float, top: float, width: float, height: float,
                   y_ticks: list[tuple[float, str]], y_min: float, y_max: float) -> None:
    bottom = top + height
    line(lines, left, top, left, bottom, COLORS["axis"])
    line(lines, left, bottom, left + width, bottom, COLORS["axis"])
    for val, label in y_ticks:
        if y_max == y_min:
            y = bottom
        else:
            y = bottom - (val - y_min) / (y_max - y_min) * height
        line(lines, left, y, left + width, y, COLORS["grid"])
        text(lines, left - 9, y + 4, label, size=11, anchor="end", fill=COLORS["muted"])


def load_data() -> tuple[dict, list[dict]]:
    data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return data, rows


def runs_sorted(data: dict) -> list[dict]:
    return sorted(data["runs"], key=lambda run: run["value"])


def plot_depth_outcomes(data: dict) -> Path:
    runs = runs_sorted(data)
    depths = [run["value"] for run in runs]
    width, height = 1060, 620
    left, top, plot_w, plot_h = 92, 96, 850, 390
    bottom = top + plot_h
    lines = svg_root(
        width,
        height,
        "Outcome quality by max split depth",
        "Atom pass-rate and decomposition fidelity are single-run measurements; markers show exact-answer correctness.",
    )
    rect(lines, left - 18, top - 26, plot_w + 64, plot_h + 72, COLORS["panel"], stroke="#E5E7EB", radius=8)
    plot_area_axes(lines, left, top, plot_w, plot_h, [(i / 4, fmt_pct(i / 4)) for i in range(5)], 0, 1)

    def sx(depth: int) -> float:
        return left + (depth - min(depths)) / (max(depths) - min(depths)) * plot_w

    def sy(value: float) -> float:
        return bottom - value * plot_h

    for depth in depths:
        x = sx(depth)
        line(lines, x, bottom, x, bottom + 7, COLORS["axis"])
        text(lines, x, bottom + 25, depth, size=12, anchor="middle", fill=COLORS["muted"])

    atom_points = [(sx(run["value"]), sy(run["atom_pass_rate"])) for run in runs if run["atom_pass_rate"] is not None]
    fidelity_points = [
        (sx(run["value"]), sy(run["decomposition_fidelity"]))
        for run in runs
        if run["decomposition_fidelity"] is not None
    ]
    polyline(lines, atom_points, COLORS["atom"], width=3.0)
    polyline(lines, fidelity_points, COLORS["fidelity"], width=3.0, dash="5 4")

    for run in runs:
        x = sx(run["value"])
        correct_color = COLORS["success"] if run["overall_correct"] else COLORS["fail"]
        circle(lines, x, top - 17, 8, correct_color)
        text(lines, x, top - 31, "correct" if run["overall_correct"] else "miss", size=10, anchor="middle", fill=correct_color)
        if run["atom_pass_rate"] is not None:
            y = sy(run["atom_pass_rate"])
            circle(lines, x, y, 6, COLORS["atom"])
            label = f'{run["n_pass"]}/{run["n_gradeable"]}'
            text(lines, x, y - 12, label, size=10, anchor="middle", fill=COLORS["atom"])
        else:
            text(lines, x, bottom - 8, "no gradeable atoms", size=10, anchor="middle", fill=COLORS["muted"])
        if run["decomposition_fidelity"] is not None:
            y = sy(run["decomposition_fidelity"])
            circle(lines, x, y, 5, COLORS["fidelity"])

    text(lines, left + plot_w / 2, height - 54, "pipeline.max_split_depth", size=13, anchor="middle", weight=600)
    text(lines, 22, top + plot_h / 2, "rate", size=13, anchor="middle", weight=600, rotate=-90)
    legend(lines, left + 470, top - 42, [("atom pass-rate", COLORS["atom"]), ("decomp fidelity", COLORS["fidelity"])])
    return save_svg("depth_outcomes.svg", lines)


def draw_bar_panel(lines: list[str], x0: float, y0: float, w: float, h: float, title: str,
                   depths: list[int], values: list[float], color: str, formatter,
                   y_label: str | None = None) -> None:
    rect(lines, x0, y0, w, h, "#FFFFFF", stroke="#E5E7EB", radius=8)
    text(lines, x0 + 16, y0 + 24, title, size=14, weight=700)
    plot_left, plot_top = x0 + 58, y0 + 46
    plot_w, plot_h = w - 84, h - 96
    bottom = plot_top + plot_h
    ymax = nice_max(max(values) * 1.08)
    ticks = [(0, "0"), (ymax / 2, formatter(ymax / 2)), (ymax, formatter(ymax))]
    plot_area_axes(lines, plot_left, plot_top, plot_w, plot_h, ticks, 0, ymax)
    gap = 18
    bar_w = (plot_w - gap * (len(values) + 1)) / len(values)
    for idx, (depth, value) in enumerate(zip(depths, values)):
        x = plot_left + gap + idx * (bar_w + gap)
        bar_h = 0 if ymax == 0 else value / ymax * plot_h
        y = bottom - bar_h
        rect(lines, x, y, bar_w, bar_h, color, radius=3)
        text(lines, x + bar_w / 2, y - 7, formatter(value), size=10, anchor="middle", fill=COLORS["ink"])
        text(lines, x + bar_w / 2, bottom + 20, depth, size=11, anchor="middle", fill=COLORS["muted"])
    text(lines, plot_left + plot_w / 2, y0 + h - 18, "split depth", size=11, anchor="middle", fill=COLORS["muted"])
    if y_label:
        text(lines, x0 + 14, plot_top + plot_h / 2, y_label, size=10, anchor="middle", rotate=-90, fill=COLORS["muted"])


def plot_resource_scaling(data: dict) -> Path:
    runs = runs_sorted(data)
    depths = [run["value"] for run in runs]
    width, height = 1120, 760
    lines = svg_root(
        width,
        height,
        "Resource scaling by max split depth",
        "Costs, calls, and wall time rise quickly as the decomposition depth increases.",
    )
    panels = [
        ("Total tokens", [run["cost"]["total_tokens"] for run in runs], COLORS["tokens"], fmt_k, "tokens"),
        ("Estimated cost", [run["cost"]["usd"] for run in runs], COLORS["usd"], fmt_usd, "USD"),
        ("Elapsed time", [run["elapsed_s"] / 60 for run in runs], COLORS["elapsed"], lambda v: f"{v:.0f}m", "minutes"),
        ("LLM calls", [run["cost"]["calls"] for run in runs], COLORS["calls"], fmt_int, "calls"),
    ]
    x_positions = [48, 584]
    y_positions = [88, 414]
    for idx, (title, values, color, formatter, ylabel) in enumerate(panels):
        draw_bar_panel(
            lines,
            x_positions[idx % 2],
            y_positions[idx // 2],
            488,
            280,
            title,
            depths,
            values,
            color,
            formatter,
            ylabel,
        )
    flagged = [run for run in runs if run.get("flagged")]
    if flagged:
        note = "; ".join(f'depth {run["value"]}: {run["flagged"]}' for run in flagged)
        rect(lines, 48, 704, 1024, 32, "#FFF7ED", stroke="#FDBA74", radius=6)
        text(lines, 64, 725, f"Flagged run: {note}", size=12, fill="#9A3412", weight=600)
    return save_svg("resource_scaling.svg", lines)


def plot_token_accuracy_frontier(data: dict) -> Path:
    runs = runs_sorted(data)
    width, height = 1040, 640
    left, top, plot_w, plot_h = 112, 96, 820, 410
    bottom = top + plot_h
    lines = svg_root(
        width,
        height,
        "Token/accuracy tradeoff frontier",
        "X-axis is log-scaled total tokens; null atom pass-rate is plotted at 0 and labelled n/a.",
    )
    rect(lines, left - 18, top - 26, plot_w + 64, plot_h + 74, COLORS["panel"], stroke="#E5E7EB", radius=8)
    x_min, x_max = 12000, 800000
    log_min, log_max = math.log10(x_min), math.log10(x_max)

    def sx(tokens: int) -> float:
        return left + (math.log10(tokens) - log_min) / (log_max - log_min) * plot_w

    def sy(rate: float) -> float:
        return bottom - rate * plot_h

    plot_area_axes(lines, left, top, plot_w, plot_h, [(i / 4, fmt_pct(i / 4)) for i in range(5)], 0, 1)
    for tick in [20000, 50000, 100000, 200000, 500000]:
        x = sx(tick)
        line(lines, x, bottom, x, bottom + 7, COLORS["axis"])
        line(lines, x, top, x, bottom, COLORS["grid"])
        text(lines, x, bottom + 25, fmt_k(tick), size=11, anchor="middle", fill=COLORS["muted"])

    frontier_lookup = {item["run_tag"]: item for item in data["token_score_frontier"]}
    frontier_points = []
    for run in runs:
        if run["run_tag"] in frontier_lookup:
            rate = run["atom_pass_rate"] if run["atom_pass_rate"] is not None else 0
            frontier_points.append((sx(run["cost"]["total_tokens"]), sy(rate)))
    polyline(lines, frontier_points, COLORS["frontier"], width=2.5, dash="7 4")

    for run in runs:
        rate = run["atom_pass_rate"] if run["atom_pass_rate"] is not None else 0
        x, y = sx(run["cost"]["total_tokens"]), sy(rate)
        color = COLORS["success"] if run["overall_correct"] else COLORS["fail"]
        circle(lines, x, y, 9, color)
        text(lines, x + 13, y - 10, f'd{run["value"]}', size=12, weight=700, fill=color)
        label = fmt_pct(run["atom_pass_rate"]) if run["atom_pass_rate"] is not None else "n/a"
        text(lines, x + 13, y + 6, label, size=10, fill=COLORS["muted"])

    legend(lines, left + 450, top - 42, [("correct", COLORS["success"]), ("miss", COLORS["fail"]), ("frontier", COLORS["frontier"])])
    text(lines, left + plot_w / 2, height - 58, "total tokens, log scale", size=13, anchor="middle", weight=600)
    text(lines, 32, top + plot_h / 2, "atom pass-rate", size=13, anchor="middle", weight=600, rotate=-90)
    return save_svg("token_accuracy_frontier.svg", lines)


def plot_atom_gradeability(data: dict) -> Path:
    runs = runs_sorted(data)
    depths = [run["value"] for run in runs]
    width, height = 1080, 640
    left, top, plot_w, plot_h = 96, 96, 860, 410
    bottom = top + plot_h
    lines = svg_root(
        width,
        height,
        "Atom grading breakdown by split depth",
        "Stacked counts show how many generated atoms passed, failed, or could not be graded.",
    )
    rect(lines, left - 18, top - 26, plot_w + 68, plot_h + 76, COLORS["panel"], stroke="#E5E7EB", radius=8)
    ymax = nice_max(max(run["n_atoms"] for run in runs) * 1.05)
    ticks = [(0, "0"), (ymax / 2, fmt_int(ymax / 2)), (ymax, fmt_int(ymax))]
    plot_area_axes(lines, left, top, plot_w, plot_h, ticks, 0, ymax)
    gap = 28
    bar_w = (plot_w - gap * (len(runs) + 1)) / len(runs)

    for idx, run in enumerate(runs):
        x = left + gap + idx * (bar_w + gap)
        y_cursor = bottom
        segments = [
            ("pass", run["n_pass"], COLORS["pass"]),
            ("fail", run.get("n_fail", max(run["n_gradeable"] - run["n_pass"], 0)), COLORS["miss"]),
            ("ungradeable", run["n_ungradeable"], COLORS["ungradeable"]),
        ]
        for _, count, color in segments:
            height_seg = 0 if ymax == 0 else count / ymax * plot_h
            y_cursor -= height_seg
            rect(lines, x, y_cursor, bar_w, height_seg, color, radius=2)
        text(lines, x + bar_w / 2, y_cursor - 8, fmt_int(run["n_atoms"]), size=10, anchor="middle")
        gradeable_share = run["n_gradeable"] / run["n_atoms"] if run["n_atoms"] else 0
        text(lines, x + bar_w / 2, bottom + 21, f'd{run["value"]}', size=12, anchor="middle", fill=COLORS["muted"])
        text(lines, x + bar_w / 2, bottom + 38, f'{fmt_pct(gradeable_share)} gradeable', size=9, anchor="middle", fill=COLORS["muted"])

    legend(lines, left + 480, top - 42, [("pass", COLORS["pass"]), ("fail", COLORS["miss"]), ("ungradeable", COLORS["ungradeable"])])
    text(lines, left + plot_w / 2, height - 50, "split depth", size=13, anchor="middle", weight=600)
    text(lines, 30, top + plot_h / 2, "atom count", size=13, anchor="middle", weight=600, rotate=-90)
    return save_svg("atom_gradeability.svg", lines)


def plot_role_cost_mix(data: dict) -> Path:
    runs = runs_sorted(data)
    width, height = 1080, 640
    left, top, plot_w, plot_h = 96, 96, 860, 410
    bottom = top + plot_h
    roles = ["execution", "verification", "splitter", "combiner"]
    lines = svg_root(
        width,
        height,
        "Cost mix by model role",
        "Stacked USD estimates by role; zero-cost roles are omitted.",
    )
    rect(lines, left - 18, top - 26, plot_w + 68, plot_h + 76, COLORS["panel"], stroke="#E5E7EB", radius=8)
    totals = [sum(run["per_role_cost"].get(role, {}).get("usd", 0.0) for role in roles) for run in runs]
    ymax = nice_max(max(totals) * 1.05)
    ticks = [(0, "$0"), (ymax / 2, fmt_usd(ymax / 2)), (ymax, fmt_usd(ymax))]
    plot_area_axes(lines, left, top, plot_w, plot_h, ticks, 0, ymax)
    gap = 28
    bar_w = (plot_w - gap * (len(runs) + 1)) / len(runs)
    for idx, run in enumerate(runs):
        x = left + gap + idx * (bar_w + gap)
        y_cursor = bottom
        for role in roles:
            value = run["per_role_cost"].get(role, {}).get("usd", 0.0)
            height_seg = 0 if ymax == 0 else value / ymax * plot_h
            y_cursor -= height_seg
            if height_seg > 0:
                rect(lines, x, y_cursor, bar_w, height_seg, COLORS[role], radius=2)
        text(lines, x + bar_w / 2, y_cursor - 8, fmt_usd(totals[idx]), size=10, anchor="middle")
        text(lines, x + bar_w / 2, bottom + 24, f'd{run["value"]}', size=12, anchor="middle", fill=COLORS["muted"])
    legend(lines, left + 356, top - 42, [(role, COLORS[role]) for role in roles])
    text(lines, left + plot_w / 2, height - 52, "split depth", size=13, anchor="middle", weight=600)
    text(lines, 30, top + plot_h / 2, "USD", size=13, anchor="middle", weight=600, rotate=-90)
    return save_svg("role_cost_mix.svg", lines)


def heat_color(rate: float | None) -> str:
    if rate is None:
        return "#E5E7EB"
    if rate >= 0.85:
        return "#22C55E"
    if rate >= 0.6:
        return "#84CC16"
    if rate >= 0.35:
        return "#FACC15"
    if rate > 0:
        return "#FB923C"
    return "#FCA5A5"


def plot_level_breakdown(data: dict) -> Path:
    runs = runs_sorted(data)
    width, height = 1040, 610
    left, top = 162, 108
    cell_w, cell_h = 118, 58
    lines = svg_root(
        width,
        height,
        "Per-level atom pass-rates",
        "Each cell shows passed / gradeable atoms for that decomposition level; gray means no gradeable atoms.",
    )
    rect(lines, left - 92, top - 48, cell_w * 6 + 128, cell_h * 6 + 98, COLORS["panel"], stroke="#E5E7EB", radius=8)
    for level in range(1, 7):
        x = left + (level - 1) * cell_w
        text(lines, x + cell_w / 2, top - 18, f'level {level}', size=12, anchor="middle", weight=700, fill=COLORS["muted"])
    for row_idx, run in enumerate(runs):
        y = top + row_idx * cell_h
        text(lines, left - 18, y + cell_h / 2 + 5, f'depth {run["value"]}', size=12, anchor="end", weight=700)
        for level in range(1, 7):
            x = left + (level - 1) * cell_w
            level_info = run["levels"].get(str(level))
            if level_info is None:
                rect(lines, x + 4, y + 5, cell_w - 8, cell_h - 10, "#FFFFFF", stroke="#E5E7EB", radius=6)
                continue
            rate = level_info["pass_rate"]
            color = heat_color(rate)
            rect(lines, x + 4, y + 5, cell_w - 8, cell_h - 10, color, stroke="#FFFFFF", radius=6)
            label = "n/a" if rate is None else fmt_pct(rate)
            fill = "#172033"
            text(lines, x + cell_w / 2, y + 25, label, size=13, anchor="middle", weight=700, fill=fill)
            text(
                lines,
                x + cell_w / 2,
                y + 43,
                f'{level_info["pass"]}/{level_info["gradeable"]}, n={level_info["n"]}',
                size=10,
                anchor="middle",
                fill="#334155",
            )
    legend_y = top + cell_h * 6 + 42
    legend(lines, left, legend_y, [("0%", "#FCA5A5"), ("1-34%", "#FB923C"), ("35-59%", "#FACC15"), ("60-84%", "#84CC16"), ("85%+", "#22C55E"), ("n/a", "#E5E7EB")])
    return save_svg("level_breakdown.svg", lines)


def calculate_insights(data: dict) -> dict:
    runs = runs_sorted(data)
    successes = [run for run in runs if run["overall_correct"]]
    cheapest_success = min(successes, key=lambda run: run["cost"]["usd"]) if successes else None
    best_pass = max((run for run in runs if run["atom_pass_rate"] is not None), key=lambda run: run["atom_pass_rate"])
    highest_cost = max(runs, key=lambda run: run["cost"]["usd"])
    depth3 = next(run for run in runs if run["value"] == 3)
    depth5 = next(run for run in runs if run["value"] == 5)
    depth6 = next(run for run in runs if run["value"] == 6)
    return {
        "n_runs": len(runs),
        "n_successes": len(successes),
        "cheapest_success": cheapest_success,
        "best_pass": best_pass,
        "highest_cost": highest_cost,
        "depth3": depth3,
        "depth5": depth5,
        "depth6": depth6,
        "depth5_vs_depth3_tokens": depth5["cost"]["total_tokens"] / depth3["cost"]["total_tokens"],
        "depth5_vs_depth3_usd": depth5["cost"]["usd"] / depth3["cost"]["usd"],
        "depth6_ungradeable_share": depth6["n_ungradeable"] / depth6["n_atoms"],
        "depth5_ungradeable_share": depth5["n_ungradeable"] / depth5["n_atoms"],
    }


def write_insights(data: dict, plot_paths: list[Path]) -> Path:
    insights = calculate_insights(data)
    cheapest = insights["cheapest_success"]
    best_pass = insights["best_pass"]
    depth3 = insights["depth3"]
    depth5 = insights["depth5"]
    depth6 = insights["depth6"]
    lines = [
        "# Chess study plots and key insights",
        "",
        f"- Source summary: `{SUMMARY_JSON.relative_to(RUN_DIR)}`",
        f"- Source CSV: `{SUMMARY_CSV.relative_to(RUN_DIR)}`",
        f"- Runs analyzed: {insights['n_runs']} single-repeat runs over `pipeline.max_split_depth`.",
        "",
        "## Key insights",
        "",
        f"1. Exact-answer success appeared in {insights['n_successes']} of {insights['n_runs']} runs. The cheapest correct run was depth {cheapest['value']} at {fmt_int(cheapest['cost']['total_tokens'])} tokens, {fmt_usd(cheapest['cost']['usd'])}, and {fmt_min(cheapest['elapsed_s'])}.",
        f"2. Depth 5 had the strongest atom pass-rate ({fmt_pct(best_pass['atom_pass_rate'])}, {best_pass['n_pass']}/{best_pass['n_gradeable']} gradeable atoms), but it used {insights['depth5_vs_depth3_tokens']:.1f}x the tokens and {insights['depth5_vs_depth3_usd']:.1f}x the dollars of the depth-3 correct run.",
        f"3. Depth 4 is the clearest regression: it spent {fmt_int(278053)} tokens and {fmt_usd(1.604247)} but returned an incorrect FEN, while depth 3 solved the same problem for less.",
        f"4. Gradeability is the main bottleneck at deeper splits. Depth 5 left {depth5['n_ungradeable']}/{depth5['n_atoms']} atoms ungradeable ({fmt_pct(insights['depth5_ungradeable_share'])}); depth 6 left {depth6['n_ungradeable']}/{depth6['n_atoms']} ungradeable ({fmt_pct(insights['depth6_ungradeable_share'])}).",
        f"5. Depth 6 solved the final answer, but it was flagged `{depth6['flagged']}` with only {depth6['rollouts']} rollout, the highest cost ({fmt_usd(depth6['cost']['usd'])}), and a lower atom pass-rate ({fmt_pct(depth6['atom_pass_rate'])}) than depth 5.",
        "6. If the target metric is exact final FEN, depth 3 is the cost-conscious baseline. If the target metric is subproblem grading quality, depth 5 is the better candidate to inspect, despite the added cost.",
        "",
        "## Plots",
        "",
    ]
    for path in plot_paths:
        title = path.stem.replace("_", " ").title()
        lines.extend([f"### {title}", "", f"![{title}](plots/{path.name})", ""])
    output = RUN_DIR / "study_plots_and_insights.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def write_manifest(plot_paths: list[Path], insights_path: Path) -> Path:
    manifest = {
        "source_json": str(SUMMARY_JSON.relative_to(RUN_DIR)),
        "source_csv": str(SUMMARY_CSV.relative_to(RUN_DIR)),
        "insights_report": str(insights_path.relative_to(RUN_DIR)),
        "plots": [str(path.relative_to(RUN_DIR)) for path in plot_paths],
    }
    path = PLOT_DIR / "plot_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    data, _ = load_data()
    plot_paths = [
        plot_depth_outcomes(data),
        plot_resource_scaling(data),
        plot_token_accuracy_frontier(data),
        plot_atom_gradeability(data),
        plot_role_cost_mix(data),
        plot_level_breakdown(data),
    ]
    insights_path = write_insights(data, plot_paths)
    manifest_path = write_manifest(plot_paths, insights_path)
    print(f"Wrote {len(plot_paths)} plots")
    for path in plot_paths:
        print(path.relative_to(RUN_DIR))
    print(insights_path.relative_to(RUN_DIR))
    print(manifest_path.relative_to(RUN_DIR))


if __name__ == "__main__":
    main()
