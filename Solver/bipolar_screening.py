#!/usr/bin/env python3
"""
Bipolar Screening Script

Identifies polyhedra candidates for bipolar LED driving solutions by checking:
1. Vertex pairs (s, t) with distance >= 2
2. Edge classification into vertical (on shortest paths) and horizontal (same level)
3. Even counts for vertical edges at each level transition
4. Even counts for horizontal edges at each level
5. Full edge coverage (all edges must be vertical or horizontal)

Usage:
  python bipolar_screening.py models_test/
  python bipolar_screening.py models_test/octahedron.json
"""

import argparse
import json
import os
from collections import deque
from itertools import combinations
from typing import Dict, List, Set, Tuple

Edge = Tuple[int, int]  # (u, v) undirected: stored as (min, max)


def load_model(path: str) -> Tuple[str, int, List[Edge]]:
    """Load vertices and edges from JSON. Returns (name, n_vertices, edges)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    name = data.get("name", os.path.splitext(os.path.basename(path))[0])

    # Get vertices
    verts = data.get("vertices", data.get("Verts", data.get("V", [])))
    n = len(verts) if verts else 0

    # Get edges from various sources
    edges: Set[Edge] = set()

    raw_edges = data.get("edges", data.get("Edges", data.get("E", None)))
    if raw_edges:
        for e in raw_edges:
            u, v = int(e[0]), int(e[1])
            if u != v:
                edges.add((min(u, v), max(u, v)))
    else:
        # Derive from faces
        faces = data.get("faces", data.get("Faces", data.get("F", [])))
        for face in faces:
            k = len(face)
            for i in range(k):
                u, v = int(face[i]), int(face[(i + 1) % k])
                if u != v:
                    edges.add((min(u, v), max(u, v)))

    # Update n if edges reference higher indices
    if edges:
        max_v = max(max(e) for e in edges)
        n = max(n, max_v + 1)

    return name, n, sorted(edges)


def build_adj(n: int, edges: List[Edge]) -> Dict[int, Set[int]]:
    """Build adjacency dict from edges."""
    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def bfs_distances(adj: Dict[int, Set[int]], start: int, n: int) -> List[int]:
    """Return shortest-path distances from start (unweighted)."""
    inf = 10**9
    dist = [inf] * n
    dist[start] = 0
    queue = deque([start])
    while queue:
        v = queue.popleft()
        dv = dist[v] + 1
        for u in adj[v]:
            if dist[u] == inf:
                dist[u] = dv
                queue.append(u)
    return dist


def classify_edges(
    edges: List[Edge],
    dist_s: List[int],
    dist_t: List[int],
    d_st: int,
) -> Tuple[Dict[int, List[Edge]], Dict[int, List[Edge]], List[Edge]]:
    """
    Classify edges into vertical, horizontal, and uncovered.

    Returns:
        vertical_by_level: dict mapping level L to edges going from L to L+1
        horizontal_by_level: dict mapping level L to edges within level L
        uncovered: list of edges that are neither vertical nor horizontal
    """
    vertical_by_level: Dict[int, List[Edge]] = {}
    horizontal_by_level: Dict[int, List[Edge]] = {}
    uncovered: List[Edge] = []

    for u, v in edges:
        lu, lv = dist_s[u], dist_s[v]

        # Check if horizontal (same level)
        if lu == lv:
            if lu not in horizontal_by_level:
                horizontal_by_level[lu] = []
            horizontal_by_level[lu].append((u, v))
            continue

        # Check if vertical (on a shortest s-t path)
        # Edge (u,v) is on shortest path if dist_s[u] + 1 + dist_t[v] == d_st
        # or dist_s[v] + 1 + dist_t[u] == d_st
        on_shortest_path = (
            dist_s[u] + 1 + dist_t[v] == d_st or
            dist_s[v] + 1 + dist_t[u] == d_st
        )

        if on_shortest_path and abs(lu - lv) == 1:
            level = min(lu, lv)  # Level of the transition L -> L+1
            if level not in vertical_by_level:
                vertical_by_level[level] = []
            vertical_by_level[level].append((u, v))
        else:
            uncovered.append((u, v))

    return vertical_by_level, horizontal_by_level, uncovered


def check_bipolar_conditions(
    n: int,
    edges: List[Edge],
    adj: Dict[int, Set[int]],
    s: int,
    t: int,
) -> Dict:
    """
    Check if (s, t) pair satisfies bipolar conditions.

    Returns dict with:
        - s, t, distance
        - vertical_counts: list of edge counts per level transition
        - horizontal_counts: list of edge counts per level (excluding s and t levels)
        - covered, total: edge coverage stats
        - all_vertical_even, all_horizontal_even, all_covered: boolean checks
        - failure_reasons: list of strings explaining failures
    """
    dist_s = bfs_distances(adj, s, n)
    dist_t = bfs_distances(adj, t, n)
    d_st = dist_s[t]

    result = {
        "s": s,
        "t": t,
        "distance": d_st,
        "vertical_counts": [],
        "horizontal_counts": [],
        "covered": 0,
        "total": len(edges),
        "all_vertical_even": True,
        "all_horizontal_even": True,
        "horizontal_matches_vertical": True,
        "all_covered": True,
        "failure_reasons": [],
    }

    if d_st < 2:
        result["failure_reasons"].append("distance < 2")
        return result

    vertical, horizontal, uncovered = classify_edges(edges, dist_s, dist_t, d_st)

    # Count vertical edges per level transition (L0->L1, L1->L2, ...)
    v_counts = []
    for level in range(d_st):
        count = len(vertical.get(level, []))
        v_counts.append(count)
        if count % 2 != 0:
            result["all_vertical_even"] = False
    result["vertical_counts"] = v_counts

    # Count horizontal edges per level (excluding level 0 and level d_st)
    # Check: horizontal count must be 0 or equal to vertical count of preceding layer
    h_counts = []
    h_mismatch_levels = []
    for level in range(1, d_st):
        count = len(horizontal.get(level, []))
        h_counts.append(count)
        if count % 2 != 0:
            result["all_horizontal_even"] = False
        # Check: h_count at level L must be 0 or equal to v_counts[L-1] (edges from L-1 to L)
        v_preceding = v_counts[level - 1] if level - 1 < len(v_counts) else 0
        if count != 0 and count != v_preceding:
            result["horizontal_matches_vertical"] = False
            h_mismatch_levels.append((level, count, v_preceding))
    result["horizontal_counts"] = h_counts

    # Check coverage
    covered_count = sum(v_counts) + sum(h_counts)
    # Also count horizontal at levels 0 and d_st (if any)
    covered_count += len(horizontal.get(0, []))
    covered_count += len(horizontal.get(d_st, []))

    result["covered"] = covered_count
    result["uncovered_count"] = len(uncovered)

    if uncovered:
        result["all_covered"] = False

    # Check horizontal edge consistency (bipartite = can orient consistently)
    result["horizontal_bipartite"] = True
    non_bipartite_levels = []
    for level, h_edges in horizontal.items():
        if len(h_edges) < 2:
            continue
        # Build graph and check 2-colorability
        h_vertices = set()
        for u, v in h_edges:
            h_vertices.add(u)
            h_vertices.add(v)
        h_adj = {v: [] for v in h_vertices}
        for u, v in h_edges:
            h_adj[u].append(v)
            h_adj[v].append(u)
        # Check bipartite
        color = {}
        is_bipartite = True
        for start in h_vertices:
            if start in color:
                continue
            color[start] = 0
            stack = [start]
            while stack and is_bipartite:
                v = stack.pop()
                for u in h_adj[v]:
                    if u not in color:
                        color[u] = 1 - color[v]
                        stack.append(u)
                    elif color[u] == color[v]:
                        is_bipartite = False
                        break
        if not is_bipartite:
            result["horizontal_bipartite"] = False
            non_bipartite_levels.append(level)

    # Build failure reasons
    if not result["all_vertical_even"]:
        odd_levels = [f"L{i}" for i, c in enumerate(v_counts) if c % 2 != 0]
        result["failure_reasons"].append(f"V odd at {','.join(odd_levels)}")

    if not result["all_horizontal_even"]:
        odd_levels = [f"L{i+1}" for i, c in enumerate(h_counts) if c % 2 != 0]
        result["failure_reasons"].append(f"H odd at {','.join(odd_levels)}")

    if not result["horizontal_matches_vertical"]:
        mismatches = [f"H[L{l}]={h}!=V[L{l-1}]={v}" for l, h, v in h_mismatch_levels]
        result["failure_reasons"].append(f"H/V mismatch: {', '.join(mismatches)}")

    if not result["horizontal_bipartite"]:
        result["failure_reasons"].append(f"H not bipartite at L{','.join(str(l) for l in non_bipartite_levels)}")

    if not result["all_covered"]:
        result["failure_reasons"].append(f"{len(uncovered)} uncovered")

    return result


def screen_polyhedron(path: str) -> Dict:
    """
    Screen a polyhedron for bipolar candidates.

    Returns dict with:
        - name, n_vertices, n_edges
        - passing_pairs: list of (s, t) pairs that pass all conditions
        - best_attempt: the attempt with maximum coverage (if no passing pairs)
    """
    name, n, edges = load_model(path)
    adj = build_adj(n, edges)

    result = {
        "name": name,
        "n_vertices": n,
        "n_edges": len(edges),
        "passing_pairs": [],
        "best_attempt": None,
    }

    best_coverage = -1

    # Try all vertex pairs
    for s, t in combinations(range(n), 2):
        check = check_bipolar_conditions(n, edges, adj, s, t)

        if check["distance"] < 2:
            continue

        passes = (
            check["all_vertical_even"] and
            check["all_horizontal_even"] and
            check["horizontal_matches_vertical"] and
            check["horizontal_bipartite"] and
            check["all_covered"]
        )

        if passes:
            result["passing_pairs"].append(check)
        else:
            # Track best attempt by coverage
            if check["covered"] > best_coverage:
                best_coverage = check["covered"]
                result["best_attempt"] = check

    return result


def format_result(result: Dict) -> str:
    """Format screening result with clear layout."""
    name = result["name"]
    nv = result["n_vertices"]
    ne = result["n_edges"]

    lines = [f"{name} ({nv}V, {ne}E):"]

    if result["passing_pairs"]:
        for p in result["passing_pairs"]:
            v_str = ",".join(str(c) for c in p["vertical_counts"])
            h_str = ",".join(str(c) for c in p["horizontal_counts"]) if p["horizontal_counts"] else "-"
            lines.append(f"  PASS (s={p['s']}, t={p['t']}) d={p['distance']}  V=[{v_str}] H=[{h_str}]")
    elif result["best_attempt"]:
        p = result["best_attempt"]
        v_str = ",".join(str(c) for c in p["vertical_counts"])
        h_str = ",".join(str(c) for c in p["horizontal_counts"]) if p["horizontal_counts"] else "-"
        reasons = ", ".join(p["failure_reasons"])
        lines.append(f"  FAIL best (s={p['s']}, t={p['t']}) d={p['distance']}  V=[{v_str}] H=[{h_str}]  cover={p['covered']}/{p['total']}")
        lines.append(f"       reason: {reasons}")
    else:
        lines.append("  FAIL: no vertex pairs with distance >= 2")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Screen polyhedra for bipolar candidates")
    parser.add_argument("input", help="JSON file or folder of JSON files")
    parser.add_argument("--pass-only", "-p", action="store_true", help="Only show passing solutions")
    args = parser.parse_args()

    # Collect input files
    if os.path.isdir(args.input):
        files = sorted([
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.endswith(".json")
        ])
    else:
        files = [args.input]

    # Process each file
    passing_count = 0
    for path in files:
        result = screen_polyhedron(path)
        if result["passing_pairs"]:
            passing_count += 1
            print(format_result(result))
            print()
        elif not args.pass_only:
            print(format_result(result))
            print()

    # Summary
    print(f"=== {passing_count}/{len(files)} polyhedra have bipolar candidates ===")


if __name__ == "__main__":
    main()
