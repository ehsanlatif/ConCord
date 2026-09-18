"""Visualize the per-subproblem success trajectory of a chess_study run.

The study runner only prints a one-line headline ("8/8 pass, overall_correct
=False"). That headline is *misleading on its own*: the subproblem pass-rate
counts only the GRADEABLE atoms and silently drops the UNGRADEABLE ones, so a
run can show "8/8 pass (1.000)" while the final answer is wrong because half its
sub-problems were unfaithful decompositions that python-chess could not even
replay.

This tool reads the full per-atom grade (`grade.json`, written by chess_study
for every run) and the ground-truth trace, then answers the questions the
headline hides:

  * For EACH sub-problem (atomic unit), did it succeed against the trace?
    (PASS / FAIL / UNGRADEABLE, with the reason and the ply range it covers.)
  * What is the SUCCESS TRAJECTORY down the decomposition tree — i.e. as the
    splitter breaks a block into smaller and smaller move-chunks, does the
    per-level success climb?
  * An HONEST scorecard: gradeable pass-rate vs. *effective* score (ungradeable
    counted as not-solved) vs. ground-truth PLY COVERAGE (how many of the real
    half-moves were covered by a correctly-solved leaf).
  * STUDY mode: across a parameter sweep, how does the success score move with
    `max_split_depth` (or any swept knob) — the "does decomposition help" plot.

No third-party deps: the terminal view is ASCII, the report is a single
self-contained HTML file with inline SVG charts.

USAGE
-----
Single run (a run dir, or a grade.json directly):

    python -m concord.viz.decomp_viz \\
        --run results/chess_study/<ts>/runs/baseline_r1 \\
        --trace concord/verification/uci_to_fen_easy_6_trace.json \\
        --html results/chess_study/<ts>/runs/baseline_r1/decomp.html

Whole study (compare success vs the swept parameter):

    python -m concord.viz.decomp_viz \\
        --study results/chess_study/<ts> \\
        --trace concord/verification/uci_to_fen_easy_6_trace.json \\
        --html results/chess_study/<ts>/decomp_study.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

# Status glyphs / colors shared by the terminal and HTML views.
STATUS = {
    "PASS":        {"glyph": "✓", "color": "#1a7f37", "bg": "#dafbe1", "label": "pass"},
    "FAIL":        {"glyph": "✗", "color": "#cf222e", "bg": "#ffebe9", "label": "fail"},
    "UNGRADEABLE": {"glyph": "•", "color": "#6e7781", "bg": "#eaeef2", "label": "ungradeable"},
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_grade(run: Path) -> dict:
    """Accept a grade.json path or a run directory containing one."""
    p = run if run.is_file() else run / "grade.json"
    if not p.exists():
        raise SystemExit(f"no grade.json at {p}")
    return json.loads(p.read_text())


def infer_max_plies(grade: dict, trace: dict) -> int | None:
    """The run's gold FEN equals the trace's fen_after at exactly one ply;
    that ply IS the run's max_plies (it's how the study truncates)."""
    gold = grade.get("gold_fen")
    for s in trace.get("steps", []):
        if s["fen_after"] == gold:
            return s["ply"]
    return None


# ---------------------------------------------------------------------------
# Tree reconstruction from atom_ids
# ---------------------------------------------------------------------------

def _parent_id(atom_id: str) -> str:
    """Parent = atom_id with its last '/segment' removed."""
    return atom_id.rsplit("/", 1)[0] if "/" in atom_id else ""


def build_forest(atoms: list[dict]) -> tuple[dict[str, dict], list[str]]:
    """Return (node_by_id, root_ids). Each node gets a 'children' list of ids.

    Atoms whose parent is not itself an atom (e.g. the block root `scc_0/d0/a0`,
    which is graded separately) become forest roots.
    """
    by_id = {a["atom_id"]: {**a, "children": []} for a in atoms}
    roots: list[str] = []
    for aid in by_id:
        pid = _parent_id(aid)
        if pid in by_id:
            by_id[pid]["children"].append(aid)
        else:
            roots.append(aid)
    # stable child order by id
    for n in by_id.values():
        n["children"].sort()
    roots.sort()
    return by_id, roots


# ---------------------------------------------------------------------------
# Honest metrics: by-depth trajectory + ground-truth ply coverage
# ---------------------------------------------------------------------------

def depth_trajectory(atoms: list[dict]) -> list[dict]:
    """Per decomposition depth: counts + gradeable pass-rate + EFFECTIVE rate
    (ungradeable counted as not-solved). The success trajectory of the split."""
    by_d: dict[int, dict] = {}
    for a in atoms:
        d = a.get("depth")
        d = -1 if d is None else d
        b = by_d.setdefault(d, {"depth": d, "n": 0, "pass": 0, "fail": 0,
                                "ungradeable": 0, "faithful": 0, "faith_checked": 0})
        b["n"] += 1
        st = a["status"]
        b["pass" if st == "PASS" else "fail" if st == "FAIL" else "ungradeable"] += 1
        if a.get("slice_faithful") is not None:
            b["faith_checked"] += 1
            b["faithful"] += 1 if a["slice_faithful"] else 0
    rows = []
    for d in sorted(by_d):
        b = by_d[d]
        gradeable = b["pass"] + b["fail"]
        b["gradeable_pass_rate"] = (b["pass"] / gradeable) if gradeable else None
        b["effective_rate"] = b["pass"] / b["n"] if b["n"] else None
        b["faith_rate"] = (b["faithful"] / b["faith_checked"]
                           if b["faith_checked"] else None)
        rows.append(b)
    return rows


def ply_coverage(atoms: list[dict], max_plies: int | None) -> dict:
    """Of the run's plies, how many are covered by at least one PASS leaf whose
    faithful move_range includes them? A direct 'how much of the real problem
    did decomposition actually solve correctly' measure (0..1)."""
    if not max_plies:
        return {"max_plies": None, "covered": [], "n_covered": 0, "rate": None}
    covered = [False] * (max_plies + 1)   # index by ply (1-based); [0] unused
    for a in atoms:
        if a["status"] == "PASS" and a.get("move_range") and not a.get("has_children"):
            lo, hi = a["move_range"]
            for ply in range(lo, min(hi, max_plies) + 1):
                covered[ply] = True
    n = sum(1 for ply in range(1, max_plies + 1) if covered[ply])
    return {"max_plies": max_plies, "covered": covered,
            "n_covered": n, "rate": n / max_plies}


def scorecard(grade: dict, cov: dict) -> dict:
    n_atoms = grade.get("n_atoms", 0)
    n_pass = grade.get("n_pass", 0)
    return {
        "overall_correct": grade.get("overall_correct"),
        "overall_board_match": grade.get("overall_board_match"),
        "final_answer_fen": grade.get("final_answer_fen"),
        "gold_fen": grade.get("gold_fen"),
        "n_atoms": n_atoms,
        "n_pass": n_pass,
        "n_fail": grade.get("n_fail", 0),
        "n_ungradeable": grade.get("n_ungradeable", 0),
        "gradeable_pass_rate": grade.get("atom_pass_rate"),   # the headline number
        "effective_score": (n_pass / n_atoms) if n_atoms else None,  # honest number
        "decomposition_fidelity": grade.get("decomposition_fidelity"),
        "ply_coverage": cov["rate"],
        "n_plies_covered": cov["n_covered"],
        "max_plies": cov["max_plies"],
    }


# ---------------------------------------------------------------------------
# Terminal view
# ---------------------------------------------------------------------------

def _fmt_rate(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else " n/a "


def print_terminal(grade: dict, max_plies: int | None) -> None:
    atoms = [a for b in grade["blocks"] for a in b["atom_results"]]
    by_id, roots = build_forest(atoms)
    cov = ply_coverage(atoms, max_plies)
    sc = scorecard(grade, cov)

    print("=" * 78)
    print("DECOMPOSITION SUCCESS TRAJECTORY")
    print("=" * 78)
    print(f"  overall_correct : {sc['overall_correct']}   "
          f"(board_match={sc['overall_board_match']})")
    print(f"  final answer FEN: {sc['final_answer_fen']}")
    print(f"  gold FEN        : {sc['gold_fen']}")
    print()
    print(f"  HEADLINE  gradeable pass-rate : {_fmt_rate(sc['gradeable_pass_rate'])}"
          f"   ({sc['n_pass']}/{sc['n_pass'] + sc['n_fail']})   "
          f"<- counts only gradeable atoms")
    print(f"  HONEST    effective score     : {_fmt_rate(sc['effective_score'])}"
          f"   ({sc['n_pass']}/{sc['n_atoms']})   "
          f"<- ungradeable counted as not-solved")
    print(f"  GROUND-T  ply coverage        : {_fmt_rate(sc['ply_coverage'])}"
          + (f"   ({sc['n_plies_covered']}/{sc['max_plies']} plies solved by a "
             f"faithful PASS leaf)" if sc['max_plies'] else ""))
    print(f"            decomposition fidelity: {_fmt_rate(sc['decomposition_fidelity'])}"
          f"   ({grade.get('n_slice_faithful')}/{grade.get('n_slice_checked')})")
    print()

    # By-depth trajectory table.
    print("  PER DECOMPOSITION DEPTH (does breaking the problem down help?)")
    print(f"    {'depth':>5} | {'n':>3} | {'pass':>4} {'fail':>4} {'ungr':>4} | "
          f"{'gradeable rate':>14} | {'effective':>9} | {'faithful':>8}")
    for r in depth_trajectory(atoms):
        d = "?" if r["depth"] < 0 else r["depth"]
        print(f"    {str(d):>5} | {r['n']:>3} | {r['pass']:>4} {r['fail']:>4} "
              f"{r['ungradeable']:>4} | {_fmt_rate(r['gradeable_pass_rate']):>14} | "
              f"{_fmt_rate(r['effective_rate']):>9} | {_fmt_rate(r['faith_rate']):>8}")
    print()

    # ASCII tree.
    print("  DECOMPOSITION TREE  (✓ pass  ✗ fail  • ungradeable)")
    for rid in roots:
        _print_node(by_id, rid, prefix="    ", is_last=True)
    print()


def _print_node(by_id: dict, aid: str, prefix: str, is_last: bool) -> None:
    n = by_id[aid]
    glyph = STATUS[n["status"]]["glyph"]
    branch = "└─ " if is_last else "├─ "
    leaf = n.get("is_atomic")
    rng = n.get("move_range")
    rng_s = f"plies {rng[0]}-{rng[1]}" if rng else (
        f"start ply {n['input_ply']}" if n.get("input_ply") is not None else "ply ?")
    tag = "leaf" if leaf else "node"
    line = (f"{prefix}{branch}{glyph} d{n['depth']} {tag} "
            f"[{n['n_moves']}mv {rng_s}] {aid.split('/')[-1]}")
    if n["status"] != "PASS" and n.get("reason"):
        line += f"   <- {n['reason'][:54]}"
    print(line)
    kids = n["children"]
    child_prefix = prefix + ("   " if is_last else "│  ")
    for i, c in enumerate(kids):
        _print_node(by_id, c, child_prefix, i == len(kids) - 1)


# ---------------------------------------------------------------------------
# Inline-SVG chart helpers (no matplotlib)
# ---------------------------------------------------------------------------

def svg_stacked_depth(rows: list[dict]) -> str:
    """Stacked bar per depth: pass (green) / fail (red) / ungradeable (gray)."""
    if not rows:
        return "<p>no atoms</p>"
    W, H, pad = 520, 260, 40
    bw = (W - 2 * pad) / len(rows)
    maxn = max(r["n"] for r in rows) or 1
    sh = H - 2 * pad
    bars = []
    for i, r in enumerate(rows):
        x = pad + i * bw + bw * 0.15
        w = bw * 0.7
        y = H - pad
        for key, col in (("pass", STATUS["PASS"]["color"]),
                         ("fail", STATUS["FAIL"]["color"]),
                         ("ungradeable", STATUS["UNGRADEABLE"]["color"])):
            seg = r[key] / maxn * sh
            if seg <= 0:
                continue
            y -= seg
            bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                        f'height="{seg:.1f}" fill="{col}"><title>depth '
                        f'{r["depth"]} {key}: {r[key]}</title></rect>')
        d = "?" if r["depth"] < 0 else r["depth"]
        bars.append(f'<text x="{pad + i*bw + bw/2:.1f}" y="{H-pad+16}" '
                    f'text-anchor="middle" font-size="12">d{d}</text>')
        bars.append(f'<text x="{pad + i*bw + bw/2:.1f}" y="{y-4:.1f}" '
                    f'text-anchor="middle" font-size="11" fill="#444">'
                    f'{r["n"]}</text>')
    axis = (f'<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" '
            f'stroke="#999"/>')
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:540px">'
            f'{axis}{"".join(bars)}'
            f'<text x="{pad}" y="20" font-size="13" font-weight="600">'
            f'Atoms per decomposition depth</text></svg>')


def svg_rate_lines(rows: list[dict]) -> str:
    """Two lines vs depth: gradeable pass-rate and effective score (0..1)."""
    if not rows:
        return ""
    W, H, pad = 520, 260, 40
    plot_w, plot_h = W - 2 * pad, H - 2 * pad
    xs = [pad + (plot_w * i / max(1, len(rows) - 1)) for i in range(len(rows))]

    def to_pts(key):
        pts = []
        for x, r in zip(xs, rows):
            v = r[key]
            if v is None:
                continue
            y = H - pad - v * plot_h
            pts.append((x, y, v))
        return pts

    def polyline(pts, col):
        if not pts:
            return ""
        line = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
        dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{col}">'
                       f'<title>{v:.3f}</title></circle>' for x, y, v in pts)
        return (f'<polyline points="{line}" fill="none" stroke="{col}" '
                f'stroke-width="2.5"/>{dots}')

    gridlines = "".join(
        f'<line x1="{pad}" y1="{H-pad-f*plot_h:.1f}" x2="{W-pad}" '
        f'y2="{H-pad-f*plot_h:.1f}" stroke="#eee"/>'
        f'<text x="{pad-6}" y="{H-pad-f*plot_h+4:.1f}" text-anchor="end" '
        f'font-size="10" fill="#999">{f:.1f}</text>'
        for f in (0, 0.25, 0.5, 0.75, 1.0))
    labels = "".join(
        f'<text x="{x:.1f}" y="{H-pad+16}" text-anchor="middle" font-size="12">'
        f'd{"?" if r["depth"]<0 else r["depth"]}</text>'
        for x, r in zip(xs, rows))
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:540px">'
            f'{gridlines}{labels}'
            f'{polyline(to_pts("gradeable_pass_rate"), "#1a7f37")}'
            f'{polyline(to_pts("effective_rate"), "#bf3989")}'
            f'<text x="{pad}" y="20" font-size="13" font-weight="600">'
            f'Success rate vs depth</text>'
            f'<text x="{W-pad}" y="20" text-anchor="end" font-size="11">'
            f'<tspan fill="#1a7f37">— gradeable</tspan>  '
            f'<tspan fill="#bf3989">— effective</tspan></text></svg>')


def svg_ply_strip(cov: dict, atoms: list[dict]) -> str:
    """A timeline of plies 1..max; green where a faithful PASS leaf covered it,
    gray where no leaf correctly solved it."""
    mp = cov["max_plies"]
    if not mp:
        return ""
    W, cell = 620, max(8, min(34, (620 - 80) // mp))
    H = 70
    covered = cov["covered"]
    cells = []
    for ply in range(1, mp + 1):
        x = 60 + (ply - 1) * cell
        col = STATUS["PASS"]["bg"] if covered[ply] else STATUS["UNGRADEABLE"]["bg"]
        edge = STATUS["PASS"]["color"] if covered[ply] else STATUS["UNGRADEABLE"]["color"]
        cells.append(f'<rect x="{x}" y="28" width="{cell-2}" height="{cell-2}" '
                     f'fill="{col}" stroke="{edge}"><title>ply {ply}: '
                     f'{"covered" if covered[ply] else "NOT covered"}</title></rect>')
        if cell >= 16:
            cells.append(f'<text x="{x+(cell-2)/2:.1f}" y="{28+(cell-2)/2+4:.1f}" '
                         f'text-anchor="middle" font-size="9" fill="{edge}">'
                         f'{ply}</text>')
    return (f'<svg viewBox="0 0 {max(W, 60+mp*cell+10)} {H}" width="100%">'
            f'<text x="0" y="18" font-size="13" font-weight="600">'
            f'Ground-truth ply coverage ({cov["n_covered"]}/{mp})</text>'
            f'<text x="0" y="{28+(cell-2)/2+4:.1f}" font-size="10" fill="#999">'
            f'ply</text>{"".join(cells)}</svg>')


# ---------------------------------------------------------------------------
# HTML report (single run)
# ---------------------------------------------------------------------------

def _node_html(by_id: dict, aid: str) -> str:
    n = by_id[aid]
    s = STATUS[n["status"]]
    rng = n.get("move_range")
    rng_s = (f"plies {rng[0]}–{rng[1]}" if rng else
             (f"start ply {n['input_ply']}" if n.get("input_ply") is not None
              else "ply&nbsp;?"))
    tag = "leaf" if n.get("is_atomic") else "node"
    detail = ""
    if n["status"] == "FAIL":
        detail = (f'<div class="fen">expected <code>{html.escape(n.get("expected_fen") or "?")}</code>'
                  f'<br>answer&nbsp;&nbsp; <code>{html.escape(n.get("answer_fen") or "?")}</code></div>')
    elif n["status"] == "UNGRADEABLE":
        detail = f'<div class="reason">{html.escape(n.get("reason") or "")}</div>'
    head = (f'<div class="node" style="border-left:4px solid {s["color"]};'
            f'background:{s["bg"]}">'
            f'<span class="glyph" style="color:{s["color"]}">{s["glyph"]}</span> '
            f'<b>d{n["depth"]}</b> <span class="tag">{tag}</span> '
            f'<code class="aid">{html.escape(aid.split("/")[-1])}</code> '
            f'<span class="meta">{n["n_moves"]} moves · {rng_s}'
            + (f' · faithful' if n.get("slice_faithful") else
               (' · <span class=bad>unfaithful</span>'
                if n.get("slice_faithful") is False else '')) + '</span>'
            f'{detail}</div>')
    kids = "".join(_node_html(by_id, c) for c in n["children"])
    return f'<li>{head}{("<ul>" + kids + "</ul>") if kids else ""}</li>'


def render_html(grade: dict, max_plies: int | None, title: str) -> str:
    atoms = [a for b in grade["blocks"] for a in b["atom_results"]]
    by_id, roots = build_forest(atoms)
    cov = ply_coverage(atoms, max_plies)
    sc = scorecard(grade, cov)
    rows = depth_trajectory(atoms)

    def card(label, value, sub, good=None):
        col = ("#1a7f37" if good else "#cf222e") if good is not None else "#24292f"
        return (f'<div class="card"><div class="cval" style="color:{col}">{value}</div>'
                f'<div class="clab">{label}</div><div class="csub">{sub}</div></div>')

    pr = sc["gradeable_pass_rate"]
    eff = sc["effective_score"]
    cards = "".join([
        card("overall correct", "✓" if sc["overall_correct"] else "✗",
             "full FEN == gold", good=bool(sc["overall_correct"])),
        card("gradeable pass-rate", _fmt_rate(pr),
             f"{sc['n_pass']}/{sc['n_pass']+sc['n_fail']} · headline (drops ungradeable)"),
        card("effective score", _fmt_rate(eff),
             f"{sc['n_pass']}/{sc['n_atoms']} · ungradeable = not solved",
             good=(eff is not None and eff >= 0.999)),
        card("ply coverage", _fmt_rate(sc["ply_coverage"]),
             f"{sc['n_plies_covered']}/{sc['max_plies']} plies solved" if sc["max_plies"] else "n/a"),
        card("decomp fidelity", _fmt_rate(sc["decomposition_fidelity"]),
             f"{grade.get('n_slice_faithful')}/{grade.get('n_slice_checked')} faithful slices"),
    ])

    tree = "".join(_node_html(by_id, r) for r in roots)
    ply_strip = svg_ply_strip(cov, atoms)
    ply_section = (f'<h2>Ground-truth coverage</h2>{ply_strip}' if ply_strip else "")
    fen_cmp = ""
    if not sc["overall_correct"]:
        fen_cmp = (f'<div class="banner">Final answer is <b>wrong</b> even though '
                   f'the gradeable pass-rate is {_fmt_rate(pr)}. '
                   f'<br>gold&nbsp;&nbsp;: <code>{html.escape(sc["gold_fen"] or "")}</code>'
                   f'<br>answer: <code>{html.escape(sc["final_answer_fen"] or "")}</code></div>')

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>
body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#24292f;max-width:1000px}}
h1{{font-size:20px}} h2{{font-size:15px;margin-top:28px;border-bottom:1px solid #eaeef2;padding-bottom:4px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}
.card{{flex:1;min-width:150px;border:1px solid #d0d7de;border-radius:8px;padding:12px}}
.cval{{font-size:24px;font-weight:700}} .clab{{font-size:12px;color:#57606a;margin-top:4px}}
.csub{{font-size:11px;color:#8c959f;margin-top:2px}}
.banner{{background:#fff8c5;border:1px solid #d4a72c;border-radius:8px;padding:10px 14px;margin:8px 0}}
.charts{{display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start}}
ul{{list-style:none;padding-left:18px}} li{{margin:3px 0}}
.node{{display:inline-block;padding:4px 8px;border-radius:5px;margin:1px 0}}
.glyph{{font-weight:700}} .tag{{font-size:11px;color:#57606a;background:#fff;border:1px solid #d0d7de;border-radius:4px;padding:0 4px}}
.aid{{color:#0969da}} .meta{{font-size:12px;color:#57606a}} .bad{{color:#cf222e;font-weight:600}}
.fen,.reason{{font-size:11px;margin-top:3px}} .reason{{color:#9a6700}} code{{background:#f6f8fa;padding:1px 4px;border-radius:4px}}
table{{border-collapse:collapse;margin-top:8px}} td,th{{border:1px solid #d0d7de;padding:4px 10px;text-align:right;font-size:13px}}
th{{background:#f6f8fa}}
</style></head><body>
<h1>{html.escape(title)}</h1>
{fen_cmp}
<div class="cards">{cards}</div>
<h2>Does decomposition help? — success vs depth</h2>
<div class="charts"><div>{svg_stacked_depth(rows)}</div><div>{svg_rate_lines(rows)}</div></div>
<p style="color:#57606a;font-size:13px">As the splitter breaks the block into
smaller move-chunks (greater depth), faithful leaves cover 1–2 plies and pass
reliably. Gray segments are <b>ungradeable</b> atoms — branches the splitter gave
a position python-chess can't replay; those are where decomposition broke and
why the final answer can still be wrong.</p>
{ply_section}
<h2>Per-subproblem decomposition tree</h2>
<ul>{tree}</ul>
</body></html>"""


# ---------------------------------------------------------------------------
# STUDY mode: success vs swept parameter, across runs
# ---------------------------------------------------------------------------

def collect_study(study_dir: Path, trace: dict) -> list[dict]:
    """Load every run's grade.json under <study>/runs/*, recomputing the honest
    metrics, and pull each run's varied parameter + value from record.json."""
    out = []
    for run_dir in sorted((study_dir / "runs").glob("*")):
        gp = run_dir / "grade.json"
        rp = run_dir / "record.json"
        if not gp.exists():
            continue
        grade = json.loads(gp.read_text())
        rec = json.loads(rp.read_text()) if rp.exists() else {}
        atoms = [a for b in grade["blocks"] for a in b["atom_results"]]
        mp = infer_max_plies(grade, trace)
        cov = ply_coverage(atoms, mp)
        sc = scorecard(grade, cov)
        out.append({
            "run_tag": rec.get("run_tag", run_dir.name),
            "vary": rec.get("vary"),
            "value": rec.get("value"),
            "split_depth": (rec.get("key_params") or {}).get("pipeline.max_split_depth"),
            "total_tokens": (rec.get("cost") or {}).get("total_tokens"),
            **sc,
        })
    return out


def svg_study_lines(runs: list[dict], xkey: str, xlabel: str) -> str:
    """Lines vs the swept parameter: gradeable pass-rate, effective score,
    ply coverage, and overall-correct (0/1)."""
    pts = [r for r in runs if r.get(xkey) is not None]
    if not pts:
        return "<p>no comparable runs (need a varied parameter with grades)</p>"
    try:
        pts.sort(key=lambda r: float(r[xkey]))
    except (TypeError, ValueError):
        pts.sort(key=lambda r: str(r[xkey]))
    W, H, pad = 620, 320, 50
    plot_w, plot_h = W - 2 * pad, H - 2 * pad
    xs = [pad + (plot_w * i / max(1, len(pts) - 1)) for i in range(len(pts))]

    def line(key, col, binary=False):
        seg = []
        for x, r in zip(xs, pts):
            v = r.get(key)
            v = (1.0 if v else 0.0) if binary else v
            if v is None:
                continue
            y = H - pad - v * plot_h
            seg.append((x, y, v))
        if not seg:
            return ""
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in seg)
        dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{col}">'
                       f'<title>{v:.3f}</title></circle>' for x, y, v in seg)
        return (f'<polyline points="{poly}" fill="none" stroke="{col}" '
                f'stroke-width="2.5"/>{dots}')

    grid = "".join(
        f'<line x1="{pad}" y1="{H-pad-f*plot_h:.1f}" x2="{W-pad}" '
        f'y2="{H-pad-f*plot_h:.1f}" stroke="#eee"/>'
        f'<text x="{pad-6}" y="{H-pad-f*plot_h+4:.1f}" text-anchor="end" '
        f'font-size="10" fill="#999">{f:.2f}</text>' for f in (0, .25, .5, .75, 1))
    xlabels = "".join(
        f'<text x="{x:.1f}" y="{H-pad+18}" text-anchor="middle" font-size="11">'
        f'{html.escape(str(r[xkey]))}</text>' for x, r in zip(xs, pts))
    legend = ('<tspan fill="#1a7f37">— gradeable</tspan>  '
              '<tspan fill="#bf3989">— effective</tspan>  '
              '<tspan fill="#0969da">— ply cov</tspan>  '
              '<tspan fill="#cf222e">— overall✓</tspan>')
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:660px">'
            f'{grid}{xlabels}'
            f'{line("gradeable_pass_rate", "#1a7f37")}'
            f'{line("effective_score", "#bf3989")}'
            f'{line("ply_coverage", "#0969da")}'
            f'{line("overall_correct", "#cf222e", binary=True)}'
            f'<text x="{pad}" y="22" font-size="13" font-weight="600">'
            f'Success vs {html.escape(xlabel)}</text>'
            f'<text x="{W-pad}" y="22" text-anchor="end" font-size="10">{legend}</text>'
            f'<text x="{W/2:.0f}" y="{H-8}" text-anchor="middle" font-size="12">'
            f'{html.escape(xlabel)}</text></svg>')


def render_study_html(runs: list[dict], title: str) -> str:
    # Prefer the explicitly-swept knob; fall back to split_depth.
    varied = {r["vary"] for r in runs if r.get("vary")}
    xkey, xlabel = ("value", next(iter(varied))) if len(varied) == 1 else \
                   ("split_depth", "pipeline.max_split_depth")
    chart = svg_study_lines(runs, xkey, xlabel)
    head = ("<tr><th>run</th><th>varied</th><th>value</th><th>split depth</th>"
            "<th>overall</th><th>gradeable</th><th>effective</th>"
            "<th>ply cov</th><th>fidelity</th><th>tokens</th></tr>")
    body = "".join(
        f'<tr><td style="text-align:left">{html.escape(r["run_tag"])}</td>'
        f'<td>{html.escape(str(r.get("vary") or "—"))}</td>'
        f'<td>{html.escape(str(r.get("value") if r.get("value") is not None else "—"))}</td>'
        f'<td>{r.get("split_depth")}</td>'
        f'<td style="color:{"#1a7f37" if r["overall_correct"] else "#cf222e"}">'
        f'{"✓" if r["overall_correct"] else "✗"}</td>'
        f'<td>{_fmt_rate(r["gradeable_pass_rate"])}</td>'
        f'<td>{_fmt_rate(r["effective_score"])}</td>'
        f'<td>{_fmt_rate(r["ply_coverage"])}</td>'
        f'<td>{_fmt_rate(r["decomposition_fidelity"])}</td>'
        f'<td>{r.get("total_tokens")}</td></tr>' for r in runs)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;max-width:900px;color:#24292f}}
h1{{font-size:20px}} table{{border-collapse:collapse;margin-top:12px}}
td,th{{border:1px solid #d0d7de;padding:4px 10px;text-align:right;font-size:13px}} th{{background:#f6f8fa}}</style>
</head><body><h1>{html.escape(title)}</h1>
<p style="color:#57606a">Each point is a run. The <b>effective score</b> and
<b>ply coverage</b> lines are the honest signal — they fall when a deeper/shallower
split produces unfaithful branches, unlike the gradeable pass-rate which hides them.</p>
{chart}<h2 style="font-size:15px">Per-run</h2><table>{head}{body}</table></body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=Path, help="a run dir or grade.json (single run)")
    g.add_argument("--study", type=Path, help="a chess_study output dir (compare runs)")
    p.add_argument("--trace", type=Path,
                   default=Path("concord/verification/uci_to_fen_easy_6_trace.json"))
    p.add_argument("--max-plies", type=int, default=None,
                   help="override; else inferred by matching gold FEN to the trace")
    p.add_argument("--html", type=Path, default=None, help="write an HTML report here")
    args = p.parse_args()

    trace = json.loads(args.trace.read_text())

    if args.study:
        runs = collect_study(args.study, trace)
        if not runs:
            raise SystemExit(f"no graded runs under {args.study}/runs/*")
        print(f"[decomp_viz] {len(runs)} graded runs in {args.study}")
        for r in sorted(runs, key=lambda r: (r.get("split_depth") or 0)):
            print(f"  {r['run_tag']:24} split_depth={r.get('split_depth')} "
                  f"overall={'✓' if r['overall_correct'] else '✗'} "
                  f"gradeable={_fmt_rate(r['gradeable_pass_rate'])} "
                  f"effective={_fmt_rate(r['effective_score'])} "
                  f"ply_cov={_fmt_rate(r['ply_coverage'])}")
        if args.html:
            args.html.write_text(render_study_html(runs, f"Chess study — {args.study.name}"))
            print(f"[decomp_viz] HTML -> {args.html}")
        return

    grade = load_grade(args.run)
    max_plies = args.max_plies or infer_max_plies(grade, trace)
    print_terminal(grade, max_plies)
    if args.html:
        title = f"Decomposition trajectory — {args.run.name if args.run.is_dir() else args.run.parent.name}"
        args.html.write_text(render_html(grade, max_plies, title))
        print(f"[decomp_viz] HTML -> {args.html}")


if __name__ == "__main__":
    main()
