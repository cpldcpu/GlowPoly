#!/usr/bin/env python3
"""
Bidirectional Path Decomposition Solver

Finds a partition of all edges into edge-disjoint paths where:
  1. All paths have the SAME length L
  2. All paths share the SAME two endpoint vertices (s, t)
  3. Paths can go in EITHER direction (s→t or t→s) - supports AC driving
  4. Each edge is covered exactly once across all paths

This solves cases like the octahedron L=3 solution:
  - Path 1: 0 → 5 → 3 → 1 (0=A, 1=C)
  - Path 2: 0 → 4 → 2 → 1 (0=A, 1=C)
  - Path 3: 1 → 5 → 2 → 0 (1=A, 0=C)
  - Path 4: 1 → 4 → 3 → 0 (1=A, 0=C)

Usage:
  python bidirectional_path_decomp.py model.json
  python bidirectional_path_decomp.py models_folder/ --out results/
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple
from itertools import combinations

Edge = Tuple[int, int]  # (u, v) undirected: stored as (min, max)


def load_model(path: str) -> Tuple[int, List[Edge]]:
    """Load vertices and edges from JSON. Returns (n_vertices, edges)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

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

    return n, sorted(edges)


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


def iter_endpoint_pairs_for_length(
    vertices: List[int],
    degrees: List[int],
    odd_vertices: List[int],
    num_paths_needed: int,
):
    """
    Generate (s, t) endpoint pairs that pass degree/parity constraints.
    For a full edge cover into s-t paths, internal vertices must have even degree.

    For bidirectional paths (half forward, half reverse), endpoints can have
    degree = num_paths_needed (unidirectional) OR degree = num_paths_needed/2 (balanced bidirectional).
    """
    if len(odd_vertices) == 2:
        s, t = odd_vertices
        if degrees[s] != num_paths_needed or degrees[t] != num_paths_needed:
            return iter(())
        return iter(((s, t),))
    if len(odd_vertices) == 0:
        if num_paths_needed % 2 == 1:
            return iter(())
        # For bidirectional: also consider vertices with degree = num_paths_needed/2
        # (balanced bidirectional: half forward, half reverse)
        half_paths = num_paths_needed // 2
        candidates = [v for v in vertices if degrees[v] == num_paths_needed or
                      (half_paths > 0 and degrees[v] == half_paths)]
        if len(candidates) < 2:
            return iter(())
        return combinations(candidates, 2)
    return iter(())


def all_edges_on_length_L_paths(
    edges: List[Edge],
    dist_s: List[int],
    dist_t: List[int],
    length: int,
) -> bool:
    """Quick necessary check: every edge must lie on some length-L s-t path."""
    for u, v in edges:
        if min(dist_s[u] + 1 + dist_t[v], dist_s[v] + 1 + dist_t[u]) > length:
            return False
    return True


def enumerate_simple_paths(
    adj: Dict[int, Set[int]], 
    start: int, 
    end: int, 
    length: int,
    max_paths: int = 100000
) -> List[List[int]]:
    """
    Enumerate all simple paths of exactly 'length' edges from start to end.
    Returns list of paths, where each path is a list of vertices.
    """
    if length < 1:
        return []
    if start == end:
        return []  # No self-loops for paths
    
    found: List[List[int]] = []
    
    # DFS: (current_vertex, path, visited)
    stack: List[Tuple[int, List[int], Set[int]]] = [(start, [start], {start})]
    
    while stack and len(found) < max_paths:
        v, path, visited = stack.pop()
        
        if len(path) == length + 1:  # path has length+1 vertices for length edges
            if v == end:
                found.append(path)
            continue
        
        if len(path) > length + 1:
            continue
        
        for u in adj[v]:
            if u in visited:
                continue
            # Can we still reach end in remaining steps?
            remaining = length - (len(path) - 1) - 1
            if remaining < 0:
                continue
            stack.append((u, path + [u], visited | {u}))
    
    return found


def enumerate_simple_paths_pruned(
    adj: Dict[int, Set[int]],
    start: int,
    end: int,
    length: int,
    dist_to_end: List[int],
    max_paths: int = 10000,
    start_time: float = 0,
    timeout: float = float('inf'),
) -> List[List[int]]:
    """
    Enumerate simple paths of exactly 'length' edges with distance-based pruning.
    Returns early if timeout is exceeded.
    """
    if length < 1:
        return []
    if start == end:
        return []
    if dist_to_end[start] > length:
        return []

    found: List[List[int]] = []
    stack: List[Tuple[int, List[int], Set[int]]] = [(start, [start], {start})]
    iteration_count = 0

    while stack and len(found) < max_paths:
        # Check timeout periodically
        iteration_count += 1
        if iteration_count % 100 == 0:
            if time.time() - start_time > timeout:
                return found  # Return what we have so far
        
        v, path, visited = stack.pop()
        steps_used = len(path) - 1
        remaining = length - steps_used

        if remaining == 0:
            if v == end:
                found.append(path)
            continue

        if dist_to_end[v] > remaining:
            continue

        for u in adj[v]:
            if u in visited:
                continue
            if dist_to_end[u] > remaining - 1:
                continue
            stack.append((u, path + [u], visited | {u}))

    return found


def path_to_edges(path: List[int]) -> Set[Edge]:
    """Convert a path (list of vertices) to a set of undirected edges."""
    edges = set()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edges.add((min(u, v), max(u, v)))
    return edges


def path_to_directed_edges(path: List[int]) -> List[Tuple[int, int]]:
    """Convert a path to directed edges following vertex order."""
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def validate_no_short_circuits(
    paths: List[List[int]],
    directions: List[str],
    adj: Dict[int, Set[int]] = None,
    n: int = 0,
) -> Tuple[bool, List[str]]:
    """
    Validate that a solution has no short circuits with directed edges.

    Checks for:
    1. Non-monotonic paths (paths that backtrack in level = distance from s)
    2. Shared intermediate vertices between same-phase paths
    3. Edge shortcuts that bypass parts of other paths

    For directed edges (like LEDs):
    - Forward phase: all forward paths driven together (s=high, t=low)
    - Reverse phase: all reverse paths driven together (t=high, s=low)

    Returns (is_valid, list_of_issues).
    """
    issues = []

    if not paths:
        return (True, [])

    # Get endpoints from first path
    s = paths[0][0] if directions[0] == 'forward' else paths[0][-1]
    t = paths[0][-1] if directions[0] == 'forward' else paths[0][0]

    # Compute distances from s if adj provided
    dist_s = None
    if adj is not None and n > 0:
        inf = 10**9
        dist_s = [inf] * n
        dist_s[s] = 0
        queue = deque([s])
        while queue:
            v = queue.popleft()
            for u in adj[v]:
                if dist_s[u] == inf:
                    dist_s[u] = dist_s[v] + 1
                    queue.append(u)

    # Check 1: Non-monotonic paths (must have strictly increasing/decreasing levels)
    if dist_s is not None:
        for pi, (path, direction) in enumerate(zip(paths, directions)):
            levels = [dist_s[v] for v in path]

            if direction == 'forward':
                # Forward path: levels should be non-decreasing (can have horizontal edges)
                for j in range(len(levels) - 1):
                    if levels[j] > levels[j + 1]:
                        issues.append(
                            f"path {pi}: non-monotonic at {path[j]}→{path[j+1]} "
                            f"(level {levels[j]}→{levels[j+1]})"
                        )
                        break
            else:
                # Reverse path: levels should be non-increasing
                for j in range(len(levels) - 1):
                    if levels[j] < levels[j + 1]:
                        issues.append(
                            f"path {pi}: non-monotonic at {path[j]}→{path[j+1]} "
                            f"(level {levels[j]}→{levels[j+1]})"
                        )
                        break

    # Group paths by direction
    fwd_paths = [(i, paths[i]) for i, d in enumerate(directions) if d == 'forward']
    rev_paths = [(i, paths[i]) for i, d in enumerate(directions) if d == 'reverse']

    def check_phase_shorts(phase_paths: List[Tuple[int, List[int]]], phase_name: str) -> List[str]:
        """Check for shorts within a single phase (all paths driven together)."""
        phase_issues = []

        if len(phase_paths) < 2:
            return phase_issues

        # Check 2: Shared intermediate vertices (creates current branching)
        for i, (pi, path_i) in enumerate(phase_paths):
            intermediates_i = set(path_i[1:-1])
            for j, (pj, path_j) in enumerate(phase_paths):
                if i >= j:
                    continue
                intermediates_j = set(path_j[1:-1])
                shared = intermediates_i & intermediates_j
                if shared:
                    phase_issues.append(
                        f"{phase_name}: paths {pi} and {pj} share intermediate vertices {shared}"
                    )

        # Check 3: Edge shortcuts
        all_directed_edges = []
        for pi, path in phase_paths:
            for i in range(len(path) - 1):
                all_directed_edges.append((pi, path[i], path[i + 1]))

        for pi, path in phase_paths:
            vertex_pos = {v: pos for pos, v in enumerate(path)}

            for pj, edge_u, edge_v in all_directed_edges:
                if pi == pj:
                    continue

                if edge_u in vertex_pos and edge_v in vertex_pos:
                    pos_u = vertex_pos[edge_u]
                    pos_v = vertex_pos[edge_v]

                    if pos_u < pos_v and (pos_v - pos_u) > 1:
                        phase_issues.append(
                            f"{phase_name}: edge {edge_u}→{edge_v} (path {pj}) "
                            f"shorts path {pi} (skips {pos_v - pos_u - 1} vertices)"
                        )

        return phase_issues

    issues.extend(check_phase_shorts(fwd_paths, "forward"))
    issues.extend(check_phase_shorts(rev_paths, "reverse"))

    return (len(issues) == 0, issues)


def analyze_current_flow(
    paths: List[List[int]],
    all_edges: List[Edge],
    n_vertices: int,
) -> Dict[str, Any]:
    """
    Analyze current flow through edges when paths are driven.
    
    For each path driven individually (time-multiplexed), computes
    the current through each edge assuming unit current injection.
    
    When multiple paths with the same direction are driven simultaneously,
    current branches at vertices according to Kirchhoff's laws.
    
    Returns:
        Dict with:
        - edge_currents: normalized current per edge (avg over all paths)
        - min_current: minimum edge current
        - max_current: maximum edge current  
        - uniformity: 1 - (std/mean), 1.0 = perfect uniformity
        - brightness_ratio: min/max current ratio
    """
    from collections import defaultdict
    
    # For time-multiplexed driving (one path at a time), each edge
    # on a path gets full current (1.0) when that path is active.
    # Average brightness per edge = (times edge is used) / (total paths)
    # Since each edge appears in exactly one path, avg = 1 / num_paths
    
    # For simultaneous driving of same-direction paths:
    # Current branches at vertices based on path structure
    
    # Simple model: time-multiplexed driving
    # Each edge gets current for 1/num_paths of the time
    num_paths = len(paths)
    if num_paths == 0:
        return {
            "edge_currents": {},
            "min_current": 0.0,
            "max_current": 0.0,
            "uniformity": 0.0,
            "brightness_ratio": 0.0,
            "driving_mode": "none",
        }
    
    # Each edge appears in exactly ONE path (exact cover)
    # So each edge gets 1/num_paths duty cycle
    # This gives PERFECT uniformity for time-multiplexed driving
    
    edge_currents = {e: 1.0 / num_paths for e in all_edges}
    
    return {
        "edge_currents": {f"{e[0]}-{e[1]}": c for e, c in edge_currents.items()},
        "min_current": 1.0 / num_paths,
        "max_current": 1.0 / num_paths,
        "uniformity": 1.0,  # Perfect for time-multiplexed
        "brightness_ratio": 1.0,
        "driving_mode": "time_multiplexed",
        "duty_cycle": 1.0 / num_paths,
        "num_phases": num_paths,
    }


def analyze_simultaneous_current_flow(
    paths: List[List[int]],
    directions: List[str],
    all_edges: List[Edge],
    n_vertices: int,
) -> Dict[str, Any]:
    """
    Analyze current flow when same-direction paths are driven simultaneously.
    
    In AC driving mode:
    - All 'forward' paths are driven together (phase 1)
    - All 'reverse' paths are driven together (phase 2)
    
    When multiple paths share vertices, current branches.
    Uses Kirchhoff's laws to compute edge currents.
    
    Returns detailed current analysis per phase.
    """
    from collections import defaultdict
    import numpy as np
    
    # Separate paths by direction
    forward_paths = [p for p, d in zip(paths, directions) if d == 'forward']
    reverse_paths = [p for p, d in zip(paths, directions) if d == 'reverse']
    
    def compute_phase_currents(phase_paths: List[List[int]], phase_name: str) -> Dict[Edge, float]:
        """Compute current through each edge for a set of simultaneously-driven paths."""
        if not phase_paths:
            return {}
        
        # Build graph of edges used in this phase
        phase_edges = set()
        for p in phase_paths:
            phase_edges |= path_to_edges(p)
        
        # Build directed edge flow from paths
        # Each path contributes unit current from start to end
        directed_flow = defaultdict(float)
        for p in phase_paths:
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                # Store as (min, max) with direction sign
                e = (min(u, v), max(u, v))
                if u < v:
                    directed_flow[e] += 1.0
                else:
                    directed_flow[e] -= 1.0
        
        # For each edge, the current magnitude is the absolute flow
        # Normalized by number of paths in this phase
        edge_currents = {}
        for e in phase_edges:
            edge_currents[e] = abs(directed_flow.get(e, 0.0)) / len(phase_paths)
        
        return edge_currents
    
    # Compute currents for each phase
    forward_currents = compute_phase_currents(forward_paths, "forward")
    reverse_currents = compute_phase_currents(reverse_paths, "reverse")
    
    # Combined: average current across both phases
    # Each edge is active in exactly one phase
    all_edge_currents = {}
    num_phases = (1 if forward_paths else 0) + (1 if reverse_paths else 0)
    
    for e in all_edges:
        if e in forward_currents:
            all_edge_currents[e] = forward_currents[e] / num_phases
        elif e in reverse_currents:
            all_edge_currents[e] = reverse_currents[e] / num_phases
        else:
            all_edge_currents[e] = 0.0
    
    currents = list(all_edge_currents.values())
    if currents:
        min_c = min(currents)
        max_c = max(currents)
        mean_c = sum(currents) / len(currents)
        if mean_c > 0:
            std_c = (sum((c - mean_c) ** 2 for c in currents) / len(currents)) ** 0.5
            uniformity = max(0.0, 1.0 - std_c / mean_c)
        else:
            uniformity = 0.0
        brightness_ratio = min_c / max_c if max_c > 0 else 0.0
    else:
        min_c = max_c = uniformity = brightness_ratio = 0.0
    
    return {
        "edge_currents": {f"{e[0]}-{e[1]}": c for e, c in all_edge_currents.items()},
        "min_current": min_c,
        "max_current": max_c,
        "uniformity": round(uniformity, 4),
        "brightness_ratio": round(brightness_ratio, 4),
        "driving_mode": "simultaneous_ac",
        "forward_phase_paths": len(forward_paths),
        "reverse_phase_paths": len(reverse_paths),
        "num_phases": num_phases,
    }


def analyze_brightness_with_branching(
    paths: List[List[int]],
    directions: List[str],
    all_edges: List[Edge],
    n_vertices: int,
    adj: Dict[int, Set[int]],
) -> Dict[str, Any]:
    """
    Analyze brightness when current branches at intermediate vertices.
    
    Uses circuit analysis: when multiple paths share an intermediate vertex,
    current divides based on path resistance (assumed equal per edge).
    
    For a path from s to t with length L, if at vertex v the path can
    branch into k parallel sub-paths, current divides by 1/k.
    """
    from collections import defaultdict
    
    # Group paths by direction
    forward_paths = [p for p, d in zip(paths, directions) if d == 'forward']
    reverse_paths = [p for p, d in zip(paths, directions) if d == 'reverse']
    
    def analyze_phase(phase_paths: List[List[int]]) -> Dict[Edge, float]:
        """Analyze current distribution for paths driven simultaneously."""
        if not phase_paths:
            return {}
        
        # Build the subgraph for this phase
        phase_edges = set()
        for p in phase_paths:
            phase_edges |= path_to_edges(p)
        
        # Get start and end vertices (all paths share same endpoints)
        if not phase_paths:
            return {}
        start = phase_paths[0][0]
        end = phase_paths[0][-1]
        
        # For each vertex, count how many paths pass through it
        vertex_paths = defaultdict(list)
        for pi, p in enumerate(phase_paths):
            for v in p:
                vertex_paths[v].append(pi)
        
        # Use Kirchhoff's current law approach:
        # At each non-endpoint vertex, current in = current out
        # At endpoints: inject/sink total current = num_paths
        
        # For simple parallel paths (no internal crossings),
        # each path carries equal current = total / num_paths
        # 
        # For paths that share intermediate vertices:
        # Current at an edge depends on how many paths use it
        
        # Simple model: edge current = (paths using edge) / (num_paths)
        edge_path_count = defaultdict(int)
        for p in phase_paths:
            for e in path_to_edges(p):
                edge_path_count[e] += 1
        
        # But this overcounts! Since edges are partitioned (exact cover),
        # each edge is used by exactly 1 path, so current = 1/num_paths
        
        # The REAL branching happens when we consider the CIRCUIT:
        # All paths are parallel between s and t
        # If paths don't share intermediate edges, they're independent
        # If they share edges (which they don't in exact cover), current would branch
        
        # For exact cover: each edge is on exactly 1 path
        # When driven as parallel paths, current still = 1/num_paths per path
        edge_currents = {}
        for e in phase_edges:
            edge_currents[e] = 1.0 / len(phase_paths)
        
        return edge_currents
    
    forward_currents = analyze_phase(forward_paths)
    reverse_currents = analyze_phase(reverse_paths)
    
    # Combine phases (each edge active in one phase)
    num_phases = 2 if (forward_paths and reverse_paths) else 1
    combined_currents = {}
    
    for e in all_edges:
        if e in forward_currents:
            # Active during forward phase = 1/2 of total time (for 2-phase AC)
            combined_currents[e] = forward_currents[e] / num_phases
        elif e in reverse_currents:
            combined_currents[e] = reverse_currents[e] / num_phases
        else:
            combined_currents[e] = 0.0
    
    currents = list(combined_currents.values())
    if currents and max(currents) > 0:
        min_c = min(currents)
        max_c = max(currents)
        mean_c = sum(currents) / len(currents)
        std_c = (sum((c - mean_c) ** 2 for c in currents) / len(currents)) ** 0.5
        uniformity = max(0.0, 1.0 - std_c / mean_c) if mean_c > 0 else 0.0
        brightness_ratio = min_c / max_c if max_c > 0 else 0.0
    else:
        min_c = max_c = uniformity = brightness_ratio = 0.0
    
    return {
        "edge_currents": {f"{e[0]}-{e[1]}": round(c, 4) for e, c in combined_currents.items()},
        "min_current": round(min_c, 4),
        "max_current": round(max_c, 4),
        "avg_current": round(sum(currents) / len(currents), 4) if currents else 0.0,
        "uniformity": round(uniformity, 4),
        "brightness_ratio": round(brightness_ratio, 4),
        "forward_paths": len(forward_paths),
        "reverse_paths": len(reverse_paths),
        "num_phases": num_phases,
        "effective_duty_cycle": 1.0 / (len(paths) / num_phases) / num_phases if paths else 0.0,
    }


def analyze_simultaneous_driving_circuit(
    paths: List[List[int]],
    directions: List[str],
    all_edges: List[Edge],
    n_vertices: int,
) -> Dict[str, Any]:
    """
    Analyze current distribution when same-direction paths are driven simultaneously.
    
    Uses Kirchhoff's laws with nodal analysis to compute actual edge currents
    based on circuit resistance (assuming equal resistance per edge).
    
    In AC driving:
    - Phase 1: All forward paths driven together
    - Phase 2: All reverse paths driven together
    
    Returns brightness analysis with uniformity metrics.
    """
    from collections import defaultdict
    import numpy as np
    
    def analyze_phase_circuit(phase_paths: List[List[int]]) -> Dict[Edge, float]:
        """Compute edge currents for simultaneously-driven paths using nodal analysis."""
        if not phase_paths:
            return {}
        
        # All paths share same endpoints
        s = phase_paths[0][0]
        t = phase_paths[0][-1]
        
        # Build the subgraph used by these paths
        phase_edges = set()
        for p in phase_paths:
            phase_edges |= path_to_edges(p)
        
        # Get all vertices in subgraph
        vertices = set()
        for p in phase_paths:
            vertices.update(p)
        vertices = sorted(vertices)
        n_v = len(vertices)
        v_idx = {v: i for i, v in enumerate(vertices)}
        
        # Build conductance matrix G (assuming R=1 per edge)
        G = np.zeros((n_v, n_v))
        for e in phase_edges:
            u, v = e
            i, j = v_idx[u], v_idx[v]
            G[i, j] -= 1.0
            G[j, i] -= 1.0
            G[i, i] += 1.0
            G[j, j] += 1.0
        
        # Current injection: +1 at source, -1 at sink
        I = np.zeros(n_v)
        I[v_idx[s]] = 1.0
        I[v_idx[t]] = -1.0
        
        # Fix voltage at t = 0 (ground), solve for other node voltages
        t_idx = v_idx[t]
        mask = [i for i in range(n_v) if i != t_idx]
        G_red = G[np.ix_(mask, mask)]
        I_red = I[mask]
        
        try:
            V_red = np.linalg.solve(G_red, I_red)
        except np.linalg.LinAlgError:
            # Singular matrix - return uniform current
            return {e: 1.0 / len(phase_paths) for e in phase_edges}
        
        # Reconstruct full voltage vector
        V = np.zeros(n_v)
        j = 0
        for i in range(n_v):
            if i != t_idx:
                V[i] = V_red[j]
                j += 1
        
        # Edge currents: I = (V_u - V_v) / R, R = 1
        edge_currents = {}
        for e in phase_edges:
            u, v = e
            edge_currents[e] = abs(V[v_idx[u]] - V[v_idx[v]])
        
        return edge_currents
    
    # Separate paths by direction
    forward_paths = [p for p, d in zip(paths, directions) if d == 'forward']
    reverse_paths = [p for p, d in zip(paths, directions) if d == 'reverse']
    
    # Analyze each phase
    fwd_currents = analyze_phase_circuit(forward_paths)
    rev_currents = analyze_phase_circuit(reverse_paths)
    
    # Combine (each edge active in one phase, average over 2-phase AC)
    num_phases = (1 if forward_paths else 0) + (1 if reverse_paths else 0)
    combined = {}
    
    for e in all_edges:
        if e in fwd_currents:
            combined[e] = fwd_currents[e] / num_phases
        elif e in rev_currents:
            combined[e] = rev_currents[e] / num_phases
        else:
            combined[e] = 0.0
    
    currents = list(combined.values())
    if currents and max(currents) > 0:
        min_c = min(currents)
        max_c = max(currents)
        mean_c = sum(currents) / len(currents)
        std_c = (sum((c - mean_c) ** 2 for c in currents) / len(currents)) ** 0.5
        uniformity = max(0.0, 1.0 - std_c / mean_c) if mean_c > 0 else 0.0
        brightness_ratio = min_c / max_c if max_c > 0 else 0.0
    else:
        min_c = max_c = mean_c = uniformity = brightness_ratio = 0.0
    
    return {
        "edge_currents": {f"{e[0]}-{e[1]}": round(c, 4) for e, c in combined.items()},
        "min_current": round(min_c, 4),
        "max_current": round(max_c, 4),
        "avg_current": round(mean_c, 4),
        "uniformity": round(uniformity, 4),
        "brightness_ratio": round(brightness_ratio, 4),
        "driving_mode": "simultaneous_ac",
        "forward_paths": len(forward_paths),
        "reverse_paths": len(reverse_paths),
        "num_phases": num_phases,
    }


def solve_exact_cover_paths(
    all_edges: Set[Edge],
    paths: List[List[int]],
    path_edges: List[Set[Edge]],
    start_time: float = 0,
    timeout: float = float('inf'),
) -> Optional[List[int]]:
    """
    Find a subset of paths that covers all edges exactly once.
    Only returns solutions that have no short circuits (all paths driven simultaneously).
    Returns list of path indices, or None if impossible or timeout.
    """
    # Build edge -> candidate paths mapping
    edge_to_paths: Dict[Edge, List[int]] = {e: [] for e in all_edges}
    for pi, es in enumerate(path_edges):
        for e in es:
            if e in edge_to_paths:
                edge_to_paths[e].append(pi)

    # Quick fail: any edge with no covering path
    for e, cands in edge_to_paths.items():
        if not cands:
            return None

    # Helper to check if solution has no short circuits
    # (all paths are driven simultaneously in unidirectional mode)
    def has_short_circuit(sol: List[int]) -> bool:
        if len(sol) < 2:
            return False
        
        # Check 1: Shared intermediate vertices
        intermediates = [set(paths[pi][1:-1]) for pi in sol]
        for i in range(len(intermediates)):
            for j in range(i + 1, len(intermediates)):
                if intermediates[i] & intermediates[j]:
                    return True
        
        # Check 2: Chord edges that bypass path segments
        for pi in sol:
            path = paths[pi]
            vertex_position = {v: pos for pos, v in enumerate(path)}
            
            for pj in sol:
                if pi == pj:
                    continue
                for edge in path_edges[pj]:
                    u, v = edge
                    if u in vertex_position and v in vertex_position:
                        pos_u, pos_v = vertex_position[u], vertex_position[v]
                        if abs(pos_u - pos_v) > 1:
                            return True
        
        return False

    # Backtracking with most-constrained-edge heuristic
    uncovered = set(all_edges)
    chosen: List[int] = []
    all_solutions: List[List[int]] = []
    max_solutions = 50
    iteration_count = [0]

    def pick_edge() -> Optional[Edge]:
        best_e = None
        best_count = float('inf')
        for e in uncovered:
            count = sum(1 for pi in edge_to_paths[e] if path_edges[pi].issubset(uncovered))
            if count < best_count:
                best_count = count
                best_e = e
                if count <= 1:
                    break
        return best_e

    def backtrack() -> bool:
        # Check timeout periodically
        iteration_count[0] += 1
        if iteration_count[0] % 100 == 0:
            if time.time() - start_time > timeout:
                return True  # Timeout - stop searching
        
        if len(all_solutions) >= max_solutions:
            return True
            
        if not uncovered:
            sol = list(chosen)
            if not has_short_circuit(sol):
                all_solutions.append(sol)
                return True  # Found a perfect solution
            return False

        e = pick_edge()
        if e is None:
            return False

        for pi in edge_to_paths[e]:
            pes = path_edges[pi]
            if not pes.issubset(uncovered):
                continue

            chosen.append(pi)
            uncovered.difference_update(pes)

            if backtrack() and all_solutions:
                return True

            uncovered.update(pes)
            chosen.pop()

        return False

    backtrack()
    
    if all_solutions:
        return all_solutions[0]
    return None



def solve_exact_cover_paths_bidirectional(
    all_edges: Set[Edge],
    paths: List[List[int]],
    path_edges: List[Set[Edge]],
    path_directions: List[str],
    prefer_balanced: bool = True,
    start_time: float = 0,
    timeout: float = float('inf'),
) -> Optional[List[int]]:
    """
    Find a subset of paths that covers all edges exactly once,
    with the constraint that at least one path must be 'forward' and at least one 'reverse'.
    
    Uses backtracking with early termination when perfect solution found.
    If prefer_balanced=True, finds the best solution by vertex independence and balance.
    
    Returns list of path indices, or None if no bidirectional solution exists or timeout.
    """
    # Build edge -> candidate paths mapping
    edge_to_paths: Dict[Edge, List[int]] = {e: [] for e in all_edges}
    for pi, es in enumerate(path_edges):
        for e in es:
            if e in edge_to_paths:
                edge_to_paths[e].append(pi)

    # Quick fail: any edge with no covering path
    for e, cands in edge_to_paths.items():
        if not cands:
            return None

    # Helper to check if solution has no short circuits
    def is_perfect_solution(sol: List[int]) -> bool:
        """
        Check if solution has no short circuits.
        
        Edge directions are fixed by which path uses them.
        An edge can conduct during an operation if its orientation matches the voltage gradient.
        
        During forward operation: forward edges conduct, some forward edges might also short reverse paths
        During reverse operation: reverse edges conduct, some forward edges might also conduct (rectifier effect)
        
        A short circuit exists when:
        1. Two same-direction paths share an intermediate vertex
        2. An edge connects non-consecutive vertices on a path AND the edge would conduct during that operation
        """
        fwd_indices = [pi for pi in sol if path_directions[pi] == 'forward']
        rev_indices = [pi for pi in sol if path_directions[pi] == 'reverse']
        
        # Get directed edges from each path (preserving direction based on path order)
        fwd_directed_edges = []  # List of (u, v) where u comes before v in the forward path
        for pi in fwd_indices:
            path = paths[pi]
            for i in range(len(path) - 1):
                fwd_directed_edges.append((path[i], path[i+1]))
        
        rev_directed_edges = []  # List of (u, v) where u comes before v in the reverse path
        for pi in rev_indices:
            path = paths[pi]
            for i in range(len(path) - 1):
                rev_directed_edges.append((path[i], path[i+1]))
        
        def has_short_circuit_same_dir(indices, dir_edges):
            """Check for short circuits within same-direction paths."""
            if len(indices) < 2:
                return False
            
            # Check 1: Shared intermediate vertices
            intermediates = [set(paths[pi][1:-1]) for pi in indices]
            for i in range(len(intermediates)):
                for j in range(i + 1, len(intermediates)):
                    if intermediates[i] & intermediates[j]:
                        return True
            
            # Check 2: Chord edges from same-direction paths
            for pi in indices:
                path = paths[pi]
                vertex_position = {v: pos for pos, v in enumerate(path)}
                
                for pj in indices:
                    if pi == pj:
                        continue
                    for k in range(len(paths[pj]) - 1):
                        u, v = paths[pj][k], paths[pj][k+1]
                        if u in vertex_position and v in vertex_position:
                            pos_u, pos_v = vertex_position[u], vertex_position[v]
                            if abs(pos_u - pos_v) > 1:
                                return True
            
            return False
        
        def has_cross_direction_short(target_indices, source_directed_edges):
            """
            Check if edges from source paths create shorts on target paths.
            An edge u→v from source path creates a short on target path if:
            - Both u and v are on the target path
            - The edge direction (u→v) matches the "downhill" direction on target path
              (i.e., pos(u) < pos(v), meaning u is at higher potential)
            """
            for pi in target_indices:
                path = paths[pi]
                vertex_position = {v: pos for pos, v in enumerate(path)}
                
                for u, v in source_directed_edges:
                    if u in vertex_position and v in vertex_position:
                        pos_u, pos_v = vertex_position[u], vertex_position[v]
                        # Edge goes u→v. On target path, lower position = higher potential
                        # If pos_u < pos_v, edge goes from high to low potential, it conducts
                        if pos_u < pos_v and pos_v - pos_u > 1:
                            return True
            
            return False
        
        # Check same-direction short circuits
        if has_short_circuit_same_dir(fwd_indices, fwd_directed_edges):
            return False
        if has_short_circuit_same_dir(rev_indices, rev_directed_edges):
            return False
        
        # Check cross-direction short circuits (rectifier effect)
        # Forward edges that conduct during reverse operation
        if has_cross_direction_short(rev_indices, fwd_directed_edges):
            return False
        # Reverse edges that conduct during forward operation  
        if has_cross_direction_short(fwd_indices, rev_directed_edges):
            return False
        
        return True

    # Backtracking with early termination
    uncovered = set(all_edges)
    chosen: List[int] = []
    all_solutions: List[List[int]] = []
    max_solutions = 50
    found_perfect = [False]
    iteration_count = [0]

    def pick_edge() -> Optional[Edge]:
        best_e = None
        best_count = float('inf')
        for e in uncovered:
            count = sum(1 for pi in edge_to_paths[e] if path_edges[pi].issubset(uncovered))
            if count < best_count:
                best_count = count
                best_e = e
                if count <= 1:
                    break
        return best_e

    def backtrack() -> bool:
        # Check timeout periodically
        iteration_count[0] += 1
        if iteration_count[0] % 100 == 0:
            if time.time() - start_time > timeout:
                return True  # Timeout - stop searching
        
        if found_perfect[0] or len(all_solutions) >= max_solutions:
            return True
            
        if not uncovered:
            dirs = [path_directions[pi] for pi in chosen]
            if 'forward' in dirs and 'reverse' in dirs:
                sol = list(chosen)
                all_solutions.append(sol)
                if is_perfect_solution(sol):
                    found_perfect[0] = True
                    return True
                if not prefer_balanced:
                    return True
            return False

        e = pick_edge()
        if e is None:
            return False

        for pi in edge_to_paths[e]:
            pes = path_edges[pi]
            if not pes.issubset(uncovered):
                continue

            chosen.append(pi)
            uncovered.difference_update(pes)

            if backtrack() and (not prefer_balanced or found_perfect[0]):
                return True

            uncovered.update(pes)
            chosen.pop()

        return False

    backtrack()
    
    if not all_solutions:
        return None
    
    if found_perfect[0]:
        # Return the perfect solution (last one added)
        return all_solutions[-1]
    
    # No perfect (short-circuit-free) solution found
    # Filter to only return solutions with zero short circuits
    def count_short_circuits(sol):
        fwd = [pi for pi in sol if path_directions[pi] == 'forward']
        rev = [pi for pi in sol if path_directions[pi] == 'reverse']
        
        # Get directed edges
        fwd_directed_edges = []
        for pi in fwd:
            path = paths[pi]
            for i in range(len(path) - 1):
                fwd_directed_edges.append((path[i], path[i+1]))
        
        rev_directed_edges = []
        for pi in rev:
            path = paths[pi]
            for i in range(len(path) - 1):
                rev_directed_edges.append((path[i], path[i+1]))
        
        def count_in_group(indices):
            if len(indices) < 2:
                return 0
            
            count = 0
            
            # Count shared intermediate vertices
            intermediates = [set(paths[pi][1:-1]) for pi in indices]
            for i in range(len(intermediates)):
                for j in range(i+1, len(intermediates)):
                    count += len(intermediates[i] & intermediates[j])
            
            # Count chord edges - same-direction paths
            for pi in indices:
                path = paths[pi]
                vertex_position = {v: pos for pos, v in enumerate(path)}
                
                for pj in indices:
                    if pi == pj:
                        continue
                    for k in range(len(paths[pj]) - 1):
                        u, v = paths[pj][k], paths[pj][k+1]
                        if u in vertex_position and v in vertex_position:
                            pos_u, pos_v = vertex_position[u], vertex_position[v]
                            if abs(pos_u - pos_v) > 1:
                                count += 1
            
            return count
        
        def count_cross_direction_shorts(target_indices, source_directed_edges):
            """Count shorts from source edges on target paths (rectifier effect)."""
            count = 0
            for pi in target_indices:
                path = paths[pi]
                vertex_position = {v: pos for pos, v in enumerate(path)}
                
                for u, v in source_directed_edges:
                    if u in vertex_position and v in vertex_position:
                        pos_u, pos_v = vertex_position[u], vertex_position[v]
                        if pos_u < pos_v and pos_v - pos_u > 1:
                            count += 1
            return count
        
        total = count_in_group(fwd) + count_in_group(rev)
        # Add cross-direction shorts
        total += count_cross_direction_shorts(rev, fwd_directed_edges)
        total += count_cross_direction_shorts(fwd, rev_directed_edges)
        return total
    
    # Only return solutions with zero short circuits
    perfect_solutions = [sol for sol in all_solutions if count_short_circuits(sol) == 0]
    
    if perfect_solutions:
        # Return the most balanced one
        def balance_score(sol):
            fwd = sum(1 for pi in sol if path_directions[pi] == 'forward')
            rev = sum(1 for pi in sol if path_directions[pi] == 'reverse')
            return abs(fwd - rev)
        return min(perfect_solutions, key=balance_score)
    
    return None  # No short-circuit-free solution exists


def find_bipolar_structure(
    n: int,
    edges: List[Edge],
    adj: Dict[int, Set[int]],
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Find a valid bipolar structure (not path decomposition).

    A bipolar structure has:
    - Two poles s and t at distance >= 2
    - All edges classified as vertical (on shortest paths) or horizontal (same level)
    - Even counts for vertical edges at each level transition
    - Even counts for horizontal edges at each level
    - Horizontal count = 0 or = vertical count from preceding level

    Returns the best bipolar structure found, or None.
    """
    from itertools import combinations

    def classify_edges_bipolar(dist_s, dist_t, d_st):
        """Classify edges into vertical and horizontal."""
        vertical_by_level = {}  # level -> list of edges
        horizontal_by_level = {}  # level -> list of edges
        uncovered = []

        for u, v in edges:
            lu, lv = dist_s[u], dist_s[v]

            if lu == lv:
                # Horizontal edge at this level
                if lu not in horizontal_by_level:
                    horizontal_by_level[lu] = []
                horizontal_by_level[lu].append((u, v))
            elif abs(lu - lv) == 1:
                # Check if on shortest path
                on_path = (dist_s[u] + 1 + dist_t[v] == d_st or
                           dist_s[v] + 1 + dist_t[u] == d_st)
                if on_path:
                    level = min(lu, lv)
                    if level not in vertical_by_level:
                        vertical_by_level[level] = []
                    vertical_by_level[level].append((u, v))
                else:
                    uncovered.append((u, v))
            else:
                uncovered.append((u, v))

        return vertical_by_level, horizontal_by_level, uncovered

    def check_bipolar_valid(s, t):
        """Check if (s, t) forms a valid bipolar structure."""
        dist_s = bfs_distances(adj, s, n)
        dist_t = bfs_distances(adj, t, n)
        d_st = dist_s[t]

        if d_st < 2:
            return None

        vertical, horizontal, uncovered = classify_edges_bipolar(dist_s, dist_t, d_st)

        if uncovered:
            return None

        # Get counts
        v_counts = [len(vertical.get(l, [])) for l in range(d_st)]
        h_counts = [len(horizontal.get(l, [])) for l in range(1, d_st)]

        # Check all vertical even
        if any(c % 2 != 0 for c in v_counts):
            return None

        # Check all horizontal even
        if any(c % 2 != 0 for c in h_counts):
            return None

        # Check H matches V (H[L] == 0 or H[L] == V[L-1])
        for i, h in enumerate(h_counts):
            level = i + 1
            v_preceding = v_counts[level - 1] if level - 1 < len(v_counts) else 0
            if h != 0 and h != v_preceding:
                return None

        # Check horizontal edge consistency: edges sharing a vertex must have
        # consistent polarity (all anodes or all cathodes at that vertex)
        # For horizontal edges, we need to check if they can be oriented consistently
        for level, h_edges in horizontal.items():
            if len(h_edges) < 2:
                continue
            # Build graph of horizontal edges at this level
            # Check if they can be 2-colored (bipartite) for consistent orientation
            h_vertices = set()
            for u, v in h_edges:
                h_vertices.add(u)
                h_vertices.add(v)
            h_adj = {v: [] for v in h_vertices}
            for u, v in h_edges:
                h_adj[u].append(v)
                h_adj[v].append(u)
            # Check bipartiteness (2-colorable)
            color = {}
            for start in h_vertices:
                if start in color:
                    continue
                color[start] = 0
                stack = [start]
                while stack:
                    v = stack.pop()
                    for u in h_adj[v]:
                        if u not in color:
                            color[u] = 1 - color[v]
                            stack.append(u)
                        elif color[u] == color[v]:
                            # Odd cycle - not bipartite - can't orient consistently
                            return None

        # Valid! Build result
        return {
            "s": s,
            "t": t,
            "distance": d_st,
            "vertical_counts": v_counts,
            "horizontal_counts": h_counts,
            "vertical_edges": {l: list(es) for l, es in vertical.items()},
            "horizontal_edges": {l: list(es) for l, es in horizontal.items()},
        }

    # Try all vertex pairs
    best = None
    for s, t in combinations(range(n), 2):
        result = check_bipolar_valid(s, t)
        if result:
            if verbose:
                print(f"  Bipolar: ({s}, {t}) d={result['distance']} "
                      f"V={result['vertical_counts']} H={result['horizontal_counts']}",
                      file=sys.stderr)
            if best is None or result['distance'] < best['distance']:
                best = result

    return best


def find_bidirectional_decomposition(
    n: int,
    edges: List[Edge],
    max_L: int = 20,
    verbose: bool = False,
    algo: str = "pruned",
    timeout: float = 60.0,
) -> Optional[Dict[str, Any]]:
    """
    Find a bidirectional path decomposition where all paths share the same endpoints.
    
    Tries all vertex pairs (s, t) and path lengths L.
    Returns result dict or None, or a timeout result if time exceeded.
    """
    start_time = time.time()
    m = len(edges)
    adj = build_adj(n, edges)
    all_edges_set = set(edges)
    
    # Get all vertex pairs
    vertices = list(range(n))
    degrees = [len(adj[v]) for v in vertices]
    odd_vertices = [v for v in vertices if degrees[v] % 2 == 1]
    all_dist = None
    if algo == "pruned":
        all_dist = [bfs_distances(adj, v, n) for v in vertices]
    
    # Try different path lengths
    # For m edges with paths of length L, we need m/L paths
    # Exclude L=1 (single edges) and L=m (one giant path) - not practical solutions
    candidate_lengths = [L for L in range(2, min(m, max_L + 1)) if m % L == 0]
    
    # Prefer longer paths (fewer total paths needed)
    candidate_lengths = sorted(candidate_lengths, reverse=True)
    
    best_result = None
    
    for L in candidate_lengths:
        # Per-L timeout - each L value gets its own timeout window
        L_start_time = time.time()
        L_timeout_hit = False
        
        num_paths_needed = m // L
        if verbose:
            print(f"  Trying L={L} ({num_paths_needed} paths needed)...", file=sys.stderr, flush=True)
        
        # Try each vertex pair as the shared endpoints
        if algo == "pruned":
            endpoint_pairs = iter_endpoint_pairs_for_length(
                vertices, degrees, odd_vertices, num_paths_needed
            )
        else:
            endpoint_pairs = combinations(vertices, 2)

        for s, t in endpoint_pairs:
            # Check per-L timeout
            if time.time() - L_start_time > timeout:
                L_timeout_hit = True
                if verbose:
                    print(f"    (timeout for L={L})", file=sys.stderr, flush=True)
                break  # Move to next L value
            
            if algo == "pruned":
                dist_s = all_dist[s]
                dist_t = all_dist[t]
                if dist_s[t] > L:
                    continue
                if not all_edges_on_length_L_paths(edges, dist_s, dist_t, L):
                    continue
                paths_st = enumerate_simple_paths_pruned(adj, s, t, L, dist_t,
                    start_time=L_start_time, timeout=timeout)
                paths_ts = enumerate_simple_paths_pruned(adj, t, s, L, dist_s,
                    start_time=L_start_time, timeout=timeout)
            else:
                paths_st = enumerate_simple_paths(adj, s, t, L)
                paths_ts = enumerate_simple_paths(adj, t, s, L)
            
            if not paths_st and not paths_ts:
                continue
            
            # Combine all paths (both directions)
            all_paths = []
            path_directions = []  # 'forward' (s->t) or 'reverse' (t->s)
            
            for p in paths_st:
                all_paths.append(p)
                path_directions.append('forward')
            for p in paths_ts:
                all_paths.append(p)
                path_directions.append('reverse')
            
            # Get edge sets for each path
            path_edge_sets = [path_to_edges(p) for p in all_paths]
            
            # Filter: only keep paths using valid edges
            valid_paths = []
            valid_directions = []
            valid_edges = []
            for p, d, es in zip(all_paths, path_directions, path_edge_sets):
                if es.issubset(all_edges_set):
                    valid_paths.append(p)
                    valid_directions.append(d)
                    valid_edges.append(es)
            
            if len(valid_paths) < num_paths_needed:
                continue
            
            # Try to find bidirectional exact cover first (requires both forward and reverse paths)
            solution = None
            if paths_st and paths_ts:  # Only possible if we have paths in both directions
                solution = solve_exact_cover_paths_bidirectional(
                    all_edges_set, valid_paths, valid_edges, valid_directions,
                    start_time=L_start_time, timeout=timeout
                )
            
            # Fall back to any exact cover if no bidirectional solution
            if solution is None:
                solution = solve_exact_cover_paths(
                    all_edges_set, valid_paths, valid_edges,
                    start_time=L_start_time, timeout=timeout
                )
            
            if solution is not None:
                result_paths = [valid_paths[i] for i in solution]
                result_directions = [valid_directions[i] for i in solution]

                # Validate: check for short circuits with directed edges
                is_valid, short_issues = validate_no_short_circuits(
                    result_paths, result_directions, adj, n
                )

                if not is_valid:
                    if verbose:
                        print(f"    Rejected ({s}, {t}) L={L}: {len(short_issues)} short circuit(s)",
                              file=sys.stderr)
                        for issue in short_issues[:3]:  # Show first 3 issues
                            print(f"      - {issue}", file=sys.stderr)
                    continue  # Try next (s, t) pair

                # Count forward vs reverse paths
                forward_count = sum(1 for d in result_directions if d == 'forward')
                reverse_count = len(result_directions) - forward_count

                # Build result structure
                paths_info = []
                for pi, (p, d) in enumerate(zip(result_paths, result_directions)):
                    paths_info.append({
                        "vertices": p,
                        "direction": d,
                        "start": p[0],
                        "end": p[-1],
                        "directed_edges": path_to_directed_edges(p),
                    })

                result = {
                    "path_length": L,
                    "num_paths": len(result_paths),
                    "endpoint_pair": [s, t],
                    "forward_paths": forward_count,
                    "reverse_paths": reverse_count,
                    "is_bidirectional": forward_count > 0 and reverse_count > 0,
                    "paths": paths_info,
                }

                if verbose:
                    print(f"    Found: endpoints ({s}, {t}), {forward_count} fwd + {reverse_count} rev",
                          file=sys.stderr)

                # Return first valid solution found (longest L, first valid pair)
                return result
    
    return None


def find_all_bidirectional_decompositions(
    n: int,
    edges: List[Edge],
    max_L: int = 20,
    verbose: bool = False,
    algo: str = "pruned",
) -> List[Dict[str, Any]]:
    """
    Find ALL bidirectional path decompositions for all valid (L, s, t) combinations.
    
    Returns list of all valid decompositions.
    """
    m = len(edges)
    adj = build_adj(n, edges)
    all_edges_set = set(edges)
    vertices = list(range(n))
    degrees = [len(adj[v]) for v in vertices]
    odd_vertices = [v for v in vertices if degrees[v] % 2 == 1]
    all_dist = None
    if algo == "pruned":
        all_dist = [bfs_distances(adj, v, n) for v in vertices]
    
    candidate_lengths = [L for L in range(1, min(m + 1, max_L + 1)) if m % L == 0]
    candidate_lengths = sorted(candidate_lengths, reverse=True)
    
    all_decomps = []
    
    for L in candidate_lengths:
        num_paths_needed = m // L
        if verbose:
            print(f"  L={L} ({num_paths_needed} paths needed):", file=sys.stderr, flush=True)
        
        if algo == "pruned":
            endpoint_pairs = iter_endpoint_pairs_for_length(
                vertices, degrees, odd_vertices, num_paths_needed
            )
        else:
            endpoint_pairs = combinations(vertices, 2)

        for s, t in endpoint_pairs:
            if algo == "pruned":
                dist_s = all_dist[s]
                dist_t = all_dist[t]
                if dist_s[t] > L:
                    continue
                if not all_edges_on_length_L_paths(edges, dist_s, dist_t, L):
                    continue
                paths_st = enumerate_simple_paths_pruned(adj, s, t, L, dist_t)
                paths_ts = enumerate_simple_paths_pruned(adj, t, s, L, dist_s)
            else:
                paths_st = enumerate_simple_paths(adj, s, t, L)
                paths_ts = enumerate_simple_paths(adj, t, s, L)
            
            if not paths_st and not paths_ts:
                continue
            
            all_paths = []
            path_directions = []
            
            for p in paths_st:
                all_paths.append(p)
                path_directions.append('forward')
            for p in paths_ts:
                all_paths.append(p)
                path_directions.append('reverse')
            
            path_edge_sets = [path_to_edges(p) for p in all_paths]
            
            valid_paths = []
            valid_directions = []
            valid_edges = []
            for p, d, es in zip(all_paths, path_directions, path_edge_sets):
                if es.issubset(all_edges_set):
                    valid_paths.append(p)
                    valid_directions.append(d)
                    valid_edges.append(es)
            
            if len(valid_paths) < num_paths_needed:
                continue
            
            solution = solve_exact_cover_paths(all_edges_set, valid_paths, valid_edges)
            
            if solution is not None:
                result_paths = [valid_paths[i] for i in solution]
                result_directions = [valid_directions[i] for i in solution]
                
                forward_count = sum(1 for d in result_directions if d == 'forward')
                reverse_count = len(result_directions) - forward_count
                is_bidir = forward_count > 0 and reverse_count > 0
                
                paths_info = []
                for p, d in zip(result_paths, result_directions):
                    paths_info.append({
                        "vertices": p,
                        "direction": d,
                        "start": p[0],
                        "end": p[-1],
                        "directed_edges": path_to_directed_edges(p),
                    })
                
                result = {
                    "path_length": L,
                    "num_paths": len(result_paths),
                    "endpoint_pair": [s, t],
                    "forward_paths": forward_count,
                    "reverse_paths": reverse_count,
                    "is_bidirectional": is_bidir,
                    "paths": paths_info,
                }
                
                all_decomps.append(result)
                
                if verbose:
                    bidir = "AC" if is_bidir else "DC"
                    print(f"    ({s}, {t}): {forward_count} fwd + {reverse_count} rev [{bidir}]", 
                          file=sys.stderr, end="")
                
                # If this solution wasn't bidirectional, try to find one that is
                if not is_bidir and paths_st and paths_ts:
                    bidir_solution = solve_exact_cover_paths_bidirectional(
                        all_edges_set, valid_paths, valid_edges, valid_directions
                    )
                    if bidir_solution is not None:
                        bidir_paths = [valid_paths[i] for i in bidir_solution]
                        bidir_dirs = [valid_directions[i] for i in bidir_solution]
                        
                        fwd = sum(1 for d in bidir_dirs if d == 'forward')
                        rev = len(bidir_dirs) - fwd
                        
                        bidir_info = []
                        for p, d in zip(bidir_paths, bidir_dirs):
                            bidir_info.append({
                                "vertices": p,
                                "direction": d,
                                "start": p[0],
                                "end": p[-1],
                                "directed_edges": path_to_directed_edges(p),
                            })
                        
                        bidir_result = {
                            "path_length": L,
                            "num_paths": len(bidir_paths),
                            "endpoint_pair": [s, t],
                            "forward_paths": fwd,
                            "reverse_paths": rev,
                            "is_bidirectional": True,
                            "paths": bidir_info,
                        }
                        
                        all_decomps.append(bidir_result)
                        
                        if verbose:
                            print(f" + AC variant: {fwd} fwd + {rev} rev", file=sys.stderr, end="")
                
                if verbose:
                    print(file=sys.stderr)
    
    return all_decomps


def process_model(
    path: str,
    max_L: int = 20,
    find_all: bool = False,
    verbose: bool = False,
    algo: str = "pruned",
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Process a single model file."""
    name = os.path.basename(path)
    try:
        n, edges = load_model(path)
        
        # Calculate vertex degrees
        adj = build_adj(n, edges)
        degrees = [len(adj[v]) for v in range(n) if v in adj]
        unique_degrees = sorted(set(degrees))
        
        if find_all:
            decomps = find_all_bidirectional_decompositions(
                n, edges, max_L=max_L, verbose=verbose, algo=algo
            )
            
            if not decomps:
                return {
                    "model": name,
                    "ok": False,
                    "n_vertices": n,
                    "n_edges": len(edges),
                    "vertex_degrees": unique_degrees,
                    "message": "No bidirectional path decomposition found.",
                }
            
            # Find best (prefer bidirectional, then longest L)
            bidirectional = [d for d in decomps if d["is_bidirectional"]]
            if bidirectional:
                best = max(bidirectional, key=lambda d: d["path_length"])
            else:
                best = max(decomps, key=lambda d: d["path_length"])
            
            return {
                "model": name,
                "ok": True,
                "n_vertices": n,
                "n_edges": len(edges),
                "vertex_degrees": unique_degrees,
                "num_solutions": len(decomps),
                "num_bidirectional": len(bidirectional),
                "best_solution": best,
                "all_solutions": decomps,
            }
        else:
            result = find_bidirectional_decomposition(
                n, edges, max_L=max_L, verbose=verbose, algo=algo, timeout=timeout
            )
            
            # Check for timeout
            if result is not None and result.get("timeout"):
                return {
                    "model": name,
                    "ok": False,
                    "n_vertices": n,
                    "n_edges": len(edges),
                    "vertex_degrees": unique_degrees,
                    "message": f"Timeout after {result.get('elapsed', timeout):.1f}s",
                    "timeout": True,
                }
            
            if result is None:
                # Try bipolar structure as fallback
                bipolar = find_bipolar_structure(n, edges, adj, verbose=verbose)
                if bipolar:
                    return {
                        "model": name,
                        "ok": True,
                        "n_vertices": n,
                        "n_edges": len(edges),
                        "vertex_degrees": unique_degrees,
                        "solution_type": "bipolar",
                        "endpoint_pair": [bipolar["s"], bipolar["t"]],
                        "distance": bipolar["distance"],
                        "vertical_counts": bipolar["vertical_counts"],
                        "horizontal_counts": bipolar["horizontal_counts"],
                        "vertical_edges": bipolar["vertical_edges"],
                        "horizontal_edges": bipolar["horizontal_edges"],
                    }
                return {
                    "model": name,
                    "ok": False,
                    "n_vertices": n,
                    "n_edges": len(edges),
                    "vertex_degrees": unique_degrees,
                    "message": "No bidirectional path decomposition or bipolar structure found.",
                }

            return {
                "model": name,
                "ok": True,
                "n_vertices": n,
                "n_edges": len(edges),
                "vertex_degrees": unique_degrees,
                "solution_type": "path_decomposition",
                **result,
            }
    
    except Exception as e:
        import traceback
        return {
            "model": name,
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    ap = argparse.ArgumentParser(description="Bidirectional path decomposition solver")
    ap.add_argument("input", help="Model JSON file or folder of models")
    ap.add_argument("--out", "-o", default=None, help="Output folder (default: print to stdout)")
    ap.add_argument("--max-L", type=int, default=20, help="Maximum path length to try (default: 20)")
    ap.add_argument("--all", action="store_true", help="Find all valid decompositions, not just first")
    ap.add_argument("--verbose", "-v", action="store_true", help="Show progress")
    ap.add_argument(
        "--algo",
        choices=["baseline", "pruned"],
        default="pruned",
        help="Path search algorithm (default: pruned)",
    )
    ap.add_argument(
        "--timeout", "-t",
        type=float,
        default=60.0,
        help="Timeout in seconds per model (default: 60). Set to 0 to disable.",
    )
    args = ap.parse_args()

    # Collect input files
    if os.path.isdir(args.input):
        files = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith(".json")
        )
    else:
        files = [args.input]

    results = []
    for path in files:
        print(f"Processing {os.path.basename(path)}...", file=sys.stderr)
        result = process_model(
            path,
            max_L=args.max_L,
            find_all=args.all,
            verbose=args.verbose,
            algo=args.algo,
            timeout=args.timeout if args.timeout > 0 else float('inf'),
        )
        results.append(result)

        if result["ok"]:
            if args.all:
                print(f"  OK: {result['num_solutions']} solutions ({result['num_bidirectional']} bidirectional)",
                      file=sys.stderr)
            elif result.get('solution_type') == 'bipolar':
                print(f"  OK [BIPOLAR]: endpoints={result['endpoint_pair']}, d={result['distance']}, "
                      f"V={result['vertical_counts']}, H={result['horizontal_counts']}",
                      file=sys.stderr)
            else:
                bidir = "bidirectional (AC)" if result.get('is_bidirectional') else "unidirectional (DC)"
                print(f"  OK [PATH]: L={result['path_length']}, endpoints={result['endpoint_pair']}, {bidir}",
                      file=sys.stderr)
        else:
            print(f"  FAIL: {result.get('message', result.get('error', 'unknown'))}", file=sys.stderr)

    # Output
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        
        # Write combined results JSON
        results_path = os.path.join(args.out, "bidirectional_path_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nWrote {len(results)} result(s) to {results_path}", file=sys.stderr)
    else:
        # Single file: print result
        if len(results) == 1:
            print(json.dumps(results[0], indent=2))
        else:
            print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
