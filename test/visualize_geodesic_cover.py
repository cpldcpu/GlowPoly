#!/usr/bin/env python3
"""
visualize_geodesic_cover.py

Load geodesic cover solutions from geodesic_cover_results.json, read the
corresponding polyhedron from the models folder, and render it in 3D with one
distinct color per (s, t) pair. Each path segment is drawn with an arrow to
indicate direction.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk


def sanitize_name(name: str) -> str:
    """Lowercase and strip non-alphanumerics to match names across files."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_all_results(results_path: Path) -> List[Dict[str, Any]]:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("results file must contain a list")
    return data

def pick_result(results: List[Dict[str, Any]], poly_name: str | None) -> Dict[str, Any]:
    if not results:
        raise ValueError("results list is empty")
    if poly_name:
        target = sanitize_name(poly_name)
        for item in results:
            if sanitize_name(item.get("name", "")) == target:
                return item
        raise ValueError(f"poly '{poly_name}' not found in results")

    # Default: first with a solution, else first entry
    for item in results:
        if item.get("solution"):
            return item
    return results[0]


def find_model_file(models_dir: Path, poly_name: str) -> Path:
    target = sanitize_name(poly_name)
    candidates: List[Path] = []
    for path in models_dir.glob("*.json"):
        fname = sanitize_name(path.stem)
        if fname == target:
            return path
        # Lazy check inside file only if needed later
        candidates.append(path)

    # No filename match; inspect contents for a matching "name"
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            in_name = sanitize_name(data.get("name", ""))
            if in_name == target:
                return path
        except Exception:
            continue

    raise FileNotFoundError(f"model for '{poly_name}' not found in {models_dir}")


def load_model(models_dir: Path, poly_name: str) -> Dict[str, Any]:
    path = find_model_file(models_dir, poly_name)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "vertices" not in data or "edges" not in data:
        raise ValueError(f"{path} missing 'vertices' or 'edges'")
    return data


def extract_paths(pair: Dict[str, Any]) -> List[List[int]]:
    if "paths" in pair and pair["paths"]:
        return pair["paths"]
    if "path" in pair and pair["path"]:
        return [pair["path"]]
    return []


def edge_usage(solution: List[Dict[str, Any]]) -> Dict[tuple, List[int]]:
    """Map undirected edge (a,b) with a<=b to list of pair indices that use it."""
    usage: Dict[tuple, List[int]] = {}
    for idx, pair in enumerate(solution):
        for path in extract_paths(pair):
            for a, b in zip(path, path[1:]):
                key = (a, b) if a <= b else (b, a)
                usage.setdefault(key, []).append(idx)
    return usage


def canonical_edge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def flow_for_pair(paths: List[List[int]], s: int, t: int) -> Dict[Tuple[int, int], float]:
    """
    Distribute unit flow from s to t across the directed edges that appear in the given paths.
    Flow splits equally at branches and merges add.
    Returns flow per undirected edge (canonical).
    """
    if not paths:
        return {}

    # Build directed adjacency from path segments (dedup)
    adj: Dict[int, List[int]] = {}
    indeg: Dict[int, int] = {}
    nodes: set[int] = set()
    for p in paths:
        for a, b in zip(p, p[1:]):
            nodes.add(a); nodes.add(b)
            if b not in adj.setdefault(a, []):
                adj[a].append(b)
                indeg[b] = indeg.get(b, 0) + 1
            indeg.setdefault(a, indeg.get(a, 0))

    # Topological order (graph should be a DAG because steps increase along shortest paths)
    from collections import deque
    q = deque([n for n in nodes if indeg.get(n, 0) == 0])
    topo: List[int] = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(topo) != len(nodes):
        # Fallback: simple path-average if a cycle slipped in
        flow_edge: Dict[Tuple[int, int], float] = {}
        if not paths:
            return flow_edge
        share_per_path = 1.0 / len(paths)
        for p in paths:
            for a, b in zip(p, p[1:]):
                key = canonical_edge(a, b)
                flow_edge[key] = flow_edge.get(key, 0.0) + share_per_path
        return flow_edge

    flow_node: Dict[int, float] = {s: 1.0}
    flow_edge: Dict[Tuple[int, int], float] = {}

    for u in topo:
        out = adj.get(u, [])
        if not out:
            continue
        share = flow_node.get(u, 0.0) / len(out)
        for v in out:
            edge_key = canonical_edge(u, v)
            flow_edge[edge_key] = flow_edge.get(edge_key, 0.0) + share
            flow_node[v] = flow_node.get(v, 0.0) + share
    return flow_edge


def aggregate_flow(solution: List[Dict[str, Any]]) -> Dict[Tuple[int, int], float]:
    total: Dict[Tuple[int, int], float] = {}
    for pair in solution:
        paths = extract_paths(pair)
        s, t = int(pair["s"]), int(pair["t"])
        pf = flow_for_pair(paths, s, t)
        for e, val in pf.items():
            total[e] = total.get(e, 0.0) + val
    # Normalize so the maximum edge flow is 1.0
    if total:
        m = max(total.values())
        if m > 0:
            total = {e: v / m for e, v in total.items()}
    return total


def set_equal_aspect(ax: Any, points: Sequence[Sequence[float]]) -> None:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    max_range = max(
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs),
        1e-9,
    )
    mid_x = (max(xs) + min(xs)) * 0.5
    mid_y = (max(ys) + min(ys)) * 0.5
    mid_z = (max(zs) + min(zs)) * 0.5
    half = max_range * 0.5
    ax.set_xlim(mid_x - half, mid_x + half)
    ax.set_ylim(mid_y - half, mid_y + half)
    ax.set_zlim(mid_z - half, mid_z + half)
    ax.set_box_aspect((1, 1, 1))


def plot_poly(ax: Any, verts: List[List[float]], edges: List[Any]) -> None:
    # Light hatched-looking background (dashed thin lines)
    for e in edges:
        if isinstance(e, dict):
            u, v = e.get("source", e.get("u")), e.get("target", e.get("v"))
        else:
            u, v = e
        p0, p1 = verts[u], verts[v]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color="#bbbbbb",
            linewidth=0.8,
            alpha=0.5,
            linestyle="--",
        )

    for e in edges:
        if isinstance(e, dict):
            u, v = e.get("source", e.get("u")), e.get("target", e.get("v"))
        else:
            u, v = e
        p0, p1 = verts[u], verts[v]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color="#5555aa",
            linewidth=2.0,
            alpha=0.9,
            solid_capstyle="round",
        )
    xs, ys, zs = zip(*verts)
    ax.scatter(xs, ys, zs, color="black", s=10, alpha=0.5, edgecolors="none", zorder=3)
    # Label vertices
    for idx, (x, y, z) in enumerate(verts):
        ax.text(x, y, z, str(idx), color="#d1342b", fontsize=9, weight="bold", zorder=6)


def _orthogonal_unit(vec: Sequence[float]) -> List[float]:
    x, y, z = vec
    # Pick a non-colinear helper
    if abs(x) < 0.9:
        helper = (1.0, 0.0, 0.0)
    else:
        helper = (0.0, 1.0, 0.0)
    nx = y * helper[2] - z * helper[1]
    ny = z * helper[0] - x * helper[2]
    nz = x * helper[1] - y * helper[0]
    norm = max((nx**2 + ny**2 + nz**2) ** 0.5, 1e-9)
    return [nx / norm, ny / norm, nz / norm]


def plot_paths(ax: Any, verts: List[List[float]], solution: List[Dict[str, Any]]) -> None:
    colors = matplotlib.colormaps.get_cmap("tab20")
    usage = edge_usage(solution)
    colors_by_idx = {}
    for idx, pair in enumerate(solution):
        paths = extract_paths(pair)
        if not paths:
            continue
        color = colors(idx % colors.N)
        colors_by_idx[idx] = color
        label = f"{pair.get('s')}->{pair.get('t')}"
        added_label = False
        # Mark start/end once per pair
        s_idx, t_idx = pair.get("s"), pair.get("t")
        if s_idx is not None and t_idx is not None:
            s_pt = verts[int(s_idx)]
            t_pt = verts[int(t_idx)]
            ax.scatter([s_pt[0]], [s_pt[1]], [s_pt[2]], color=color, s=70, marker="o", edgecolors="k", linewidths=0.8)
            ax.scatter([t_pt[0]], [t_pt[1]], [t_pt[2]], color=color, s=90, marker="X", edgecolors="k", linewidths=0.8)
        for path in paths:
            for a, b in zip(path, path[1:]):
                p0, p1 = verts[a], verts[b]
                dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, linewidth=2, alpha=0.75)

                # Centered arrow to indicate direction, visible from either side.
                seg_len = max((dx**2 + dy**2 + dz**2) ** 0.5, 1e-9)
                ux, uy, uz = dx / seg_len, dy / seg_len, dz / seg_len
                arrow_len = seg_len * 0.35
                start = (
                    (p0[0] + p1[0]) * 0.5 - ux * arrow_len * 0.5,
                    (p0[1] + p1[1]) * 0.5 - uy * arrow_len * 0.5,
                    (p0[2] + p1[2]) * 0.5 - uz * arrow_len * 0.5,
                )
                key = (a, b) if a <= b else (b, a)
                idxs_for_edge = list(dict.fromkeys(usage.get(key, [idx])))
                pos = idxs_for_edge.index(idx) if idx in idxs_for_edge else 0
                offset_base = seg_len * 0.06
                normal = _orthogonal_unit((dx, dy, dz))
                offset = (pos - (len(idxs_for_edge) - 1) * 0.5) * offset_base
                start = (
                    start[0] + normal[0] * offset,
                    start[1] + normal[1] * offset,
                    start[2] + normal[2] * offset,
                )
                ax.quiver(
                    start[0], start[1], start[2],
                    ux * arrow_len, uy * arrow_len, uz * arrow_len,
                    arrow_length_ratio=0.35,
                    color=color,
                    linewidth=2.0,
                    normalize=False,
                    alpha=0.95,
                    pivot="tail",
                )
                # Add a filled triangular marker at the tip to make the head look solid.
                tip = (
                    start[0] + ux * arrow_len,
                    start[1] + uy * arrow_len,
                    start[2] + uz * arrow_len,
                )
                ax.scatter([tip[0]], [tip[1]], [tip[2]], color=color, marker="^", s=30, alpha=0.95, zorder=7)
                if not added_label:
                    ax.plot([], [], [], color=color, label=label)
                    added_label = True



def main() -> None:
    ap = argparse.ArgumentParser(description="Display geodesic cover solutions in 3D.")
    ap.add_argument("--results", default="geodesic_cover_results.json", help="Path to geodesic_cover_results.json")
    ap.add_argument("--models", default="models", help="Folder containing model JSON files")
    ap.add_argument("--poly", help="Name of the polyhedron to visualize (defaults to first with a solution)")
    args = ap.parse_args()

    results_path = Path(args.results)
    models_dir = Path(args.models)
    all_results = load_all_results(results_path)
    available = [r for r in all_results if r.get("solution")]
    if not available:
        raise SystemExit("No solutions available in results file.")
    res = pick_result(available, args.poly)
    poly_map: Dict[str, Dict[str, Any]] = {item["name"]: item for item in available}

    # Build GUI
    root = tk.Tk()
    root.title("Geodesic Cover Viewer")

    controls_frame = tk.Frame(root)
    controls_frame.pack(side=tk.TOP, fill=tk.X)

    container = tk.Frame(root)
    container.pack(fill=tk.BOTH, expand=True)

    fig = plt.Figure(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    canvas = FigureCanvasTkAgg(fig, master=container)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    stats_frame = tk.Frame(container)
    stats_frame.pack(side=tk.RIGHT, fill=tk.Y)
    stats_label = tk.Label(stats_frame, text="Edge flow", anchor="w", font=("TkDefaultFont", 10, "bold"))
    stats_label.pack(fill=tk.X, padx=4, pady=2)
    stats_text = tk.Text(stats_frame, width=28, height=20, state=tk.DISABLED)
    stats_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

    status_var = tk.StringVar()
    status_label = tk.Label(root, textvariable=status_var, anchor="w")
    status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # Dropdown for selecting polyhedron
    display_labels = {name: f"{name} (V={item['nV']}, E={item['nE']})" for name, item in poly_map.items()}
    label_to_name = {v: k for k, v in display_labels.items()}
    names_sorted = sorted(poly_map.keys(), key=lambda n: poly_map[n]["nE"])
    sel_var = tk.StringVar(value=display_labels[res["name"]])
    dropdown = tk.OptionMenu(controls_frame, sel_var, *[display_labels[n] for n in names_sorted])
    dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=2)

    def render(poly_name: str) -> None:
        r = poly_map.get(poly_name)
        if not r:
            status_var.set(f"Poly '{poly_name}' not found.")
            return
        if not r.get("solution"):
            status_var.set(f"No solution recorded for '{poly_name}'.")
            ax.clear()
            canvas.draw()
            return
        try:
            model = load_model(models_dir, r["name"])
        except Exception as e:
            status_var.set(f"Failed to load model: {e}")
            return

        verts = model["vertices"]
        edges = model["edges"]

        ax.clear()
        ax.set_title(f"{r['name']} (k={r.get('best_k')}, L={r.get('best_L')})")
        ax.set_facecolor("white")
        plot_poly(ax, verts, edges)
        plot_paths(ax, verts, r["solution"])
        set_equal_aspect(ax, verts)
        ax.view_init(elev=20, azim=30)
        ax.legend()
        ax.set_axis_off()  # scale/ticks not needed
        fig.tight_layout()
        canvas.draw()
        status_var.set(f"Showing '{poly_name}' with {len(r['solution'])} pair(s).")

        # Flow statistics
        flows = aggregate_flow(r["solution"])
        sorted_flow = sorted(flows.items(), key=lambda x: x[1], reverse=True)
        lines = [f"{u}-{v}: {val:.3f}" for (u, v), val in sorted_flow[:20]]
        stats_text.configure(state=tk.NORMAL)
        stats_text.delete("1.0", tk.END)
        stats_text.insert(tk.END, "\n".join(lines) if lines else "No flow data")
        stats_text.configure(state=tk.DISABLED)

    # Initial render
    render(res["name"])

    def on_change(*_: Any) -> None:
        selected_label = sel_var.get()
        name = label_to_name.get(selected_label, selected_label)
        render(name)

    sel_var.trace_add("write", on_change)

    root.mainloop()


if __name__ == "__main__":
    main()
