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
    """
    if len(odd_vertices) == 2:
        s, t = odd_vertices
        if degrees[s] != num_paths_needed or degrees[t] != num_paths_needed:
            return iter(())
        return iter(((s, t),))
    if len(odd_vertices) == 0:
        if num_paths_needed % 2 == 1:
            return iter(())
        candidates = [v for v in vertices if degrees[v] == num_paths_needed]
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
    max_paths: int = 100000,
) -> List[List[int]]:
    """
    Enumerate simple paths of exactly 'length' edges with distance-based pruning.
    """
    if length < 1:
        return []
    if start == end:
        return []
    if dist_to_end[start] > length:
        return []

    found: List[List[int]] = []
    stack: List[Tuple[int, List[int], Set[int]]] = [(start, [start], {start})]

    while stack and len(found) < max_paths:
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


# =============================================================================
# Dancing Links (DLX) Implementation for Exact Cover
# =============================================================================

class DLXNode:
    """Node in the Dancing Links data structure."""
    __slots__ = ['left', 'right', 'up', 'down', 'column', 'row_id']
    
    def __init__(self):
        self.left = self.right = self.up = self.down = self
        self.column = None
        self.row_id = -1


class DLXColumn(DLXNode):
    """Column header node with size tracking."""
    __slots__ = ['size', 'col_id']
    
    def __init__(self, col_id):
        super().__init__()
        self.size = 0
        self.col_id = col_id
        self.column = self


class DLX:
    """
    Dancing Links implementation of Algorithm X for exact cover.
    
    Much faster than naive backtracking for sparse matrices.
    """
    
    def __init__(self, num_columns: int):
        self.header = DLXColumn(-1)
        self.columns: List[DLXColumn] = []
        self.solution: List[int] = []
        self.all_solutions: List[List[int]] = []
        
        # Create column headers
        prev = self.header
        for i in range(num_columns):
            col = DLXColumn(i)
            col.left = prev
            col.right = self.header
            prev.right = col
            self.header.left = col
            self.columns.append(col)
            prev = col
    
    def add_row(self, row_id: int, col_indices: List[int]):
        """Add a row covering the specified columns."""
        if not col_indices:
            return
        
        first = None
        prev = None
        
        for col_idx in col_indices:
            col = self.columns[col_idx]
            
            node = DLXNode()
            node.row_id = row_id
            node.column = col
            
            # Link vertically
            node.up = col.up
            node.down = col
            col.up.down = node
            col.up = node
            col.size += 1
            
            # Link horizontally
            if first is None:
                first = node
                node.left = node.right = node
            else:
                node.left = prev
                node.right = first
                prev.right = node
                first.left = node
            
            prev = node
    
    def cover(self, col: DLXColumn):
        """Cover a column (remove from header list and all rows)."""
        col.right.left = col.left
        col.left.right = col.right
        
        row = col.down
        while row != col:
            node = row.right
            while node != row:
                node.down.up = node.up
                node.up.down = node.down
                node.column.size -= 1
                node = node.right
            row = row.down
    
    def uncover(self, col: DLXColumn):
        """Uncover a column (restore to header list and all rows)."""
        row = col.up
        while row != col:
            node = row.left
            while node != row:
                node.column.size += 1
                node.down.up = node
                node.up.down = node
                node = node.left
            row = row.up
        
        col.right.left = col
        col.left.right = col
    
    def choose_column(self) -> Optional[DLXColumn]:
        """Choose column with minimum size (S heuristic)."""
        best = None
        best_size = float('inf')
        
        col = self.header.right
        while col != self.header:
            if col.size < best_size:
                best_size = col.size
                best = col
                if best_size <= 1:
                    break
            col = col.right
        
        return best
    
    def solve(self, find_all: bool = False, max_solutions: int = 100) -> bool:
        """
        Solve the exact cover problem.
        
        Args:
            find_all: If True, find all solutions (up to max_solutions)
            max_solutions: Maximum number of solutions to find
            
        Returns:
            True if at least one solution found
        """
        if self.header.right == self.header:
            # All columns covered - found a solution!
            self.all_solutions.append(list(self.solution))
            return True
        
        if len(self.all_solutions) >= max_solutions:
            return True
        
        col = self.choose_column()
        if col is None or col.size == 0:
            return False
        
        self.cover(col)
        
        row = col.down
        while row != col:
            self.solution.append(row.row_id)
            
            # Cover all columns in this row
            node = row.right
            while node != row:
                self.cover(node.column)
                node = node.right
            
            if self.solve(find_all, max_solutions):
                if not find_all:
                    self.uncover(col)
                    return True
            
            # Uncover columns in reverse order
            self.solution.pop()
            node = row.left
            while node != row:
                self.uncover(node.column)
                node = node.left
            
            row = row.down
        
        self.uncover(col)
        return len(self.all_solutions) > 0


def solve_exact_cover_dlx(
    all_edges: List[Edge],
    path_edges: List[Set[Edge]],
    find_all: bool = False,
    max_solutions: int = 100,
) -> List[List[int]]:
    """
    Solve exact cover using Dancing Links (DLX).
    
    Args:
        all_edges: List of edges to cover
        path_edges: List of edge sets, one per path
        find_all: If True, find all solutions
        max_solutions: Maximum solutions to find
        
    Returns:
        List of solutions, where each solution is a list of path indices
    """
    # Create edge index mapping
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_cols = len(all_edges)
    
    # Build DLX matrix
    dlx = DLX(num_cols)
    
    for path_idx, edges in enumerate(path_edges):
        col_indices = [edge_to_idx[e] for e in edges if e in edge_to_idx]
        if col_indices:
            dlx.add_row(path_idx, col_indices)
    
    # Solve
    dlx.solve(find_all=find_all, max_solutions=max_solutions)
    
    return dlx.all_solutions


def solve_exact_cover_paths(
    all_edges: Set[Edge],
    paths: List[List[int]],
    path_edges: List[Set[Edge]],
) -> Optional[List[int]]:
    """
    Find a subset of paths that covers all edges exactly once.
    Returns list of path indices, or None if impossible.
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

    # Backtracking with most-constrained-edge heuristic
    uncovered = set(all_edges)
    chosen: List[int] = []

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
        if not uncovered:
            return True

        e = pick_edge()
        if e is None:
            return False

        for pi in edge_to_paths[e]:
            pes = path_edges[pi]
            if not pes.issubset(uncovered):
                continue

            chosen.append(pi)
            uncovered.difference_update(pes)

            if backtrack():
                return True

            uncovered.update(pes)
            chosen.pop()

        return False

    if backtrack():
        return chosen
    return None



def solve_exact_cover_paths_bidirectional(
    all_edges: Set[Edge],
    paths: List[List[int]],
    path_edges: List[Set[Edge]],
    path_directions: List[str],
    prefer_balanced: bool = True,
) -> Optional[List[int]]:
    """
    Find a subset of paths that covers all edges exactly once,
    with the constraint that at least one path must be 'forward' and at least one 'reverse'.
    
    Uses backtracking with early termination when perfect solution found.
    If prefer_balanced=True, finds the best solution by vertex independence and balance.
    
    Returns list of path indices, or None if no bidirectional solution exists.
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

    # Helper to check if solution has no shared intermediate vertices
    def is_perfect_solution(sol: List[int]) -> bool:
        fwd_indices = [pi for pi in sol if path_directions[pi] == 'forward']
        rev_indices = [pi for pi in sol if path_directions[pi] == 'reverse']
        
        def has_shared_intermediates(indices):
            if len(indices) < 2:
                return False
            intermediates = [set(paths[pi][1:-1]) for pi in indices]
            for i in range(len(intermediates)):
                for j in range(i + 1, len(intermediates)):
                    if intermediates[i] & intermediates[j]:
                        return True
            return False
        
        return not has_shared_intermediates(fwd_indices) and not has_shared_intermediates(rev_indices)

    # Backtracking with early termination
    uncovered = set(all_edges)
    chosen: List[int] = []
    all_solutions: List[List[int]] = []
    max_solutions = 50
    found_perfect = [False]

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
    
    # Score and return best
    def solution_score(sol):
        fwd = [pi for pi in sol if path_directions[pi] == 'forward']
        rev = [pi for pi in sol if path_directions[pi] == 'reverse']
        balance = abs(len(fwd) - len(rev))
        
        def count_shared(indices):
            if len(indices) < 2:
                return 0
            intermediates = [set(paths[pi][1:-1]) for pi in indices]
            return sum(len(intermediates[i] & intermediates[j]) 
                      for i in range(len(intermediates)) 
                      for j in range(i+1, len(intermediates)))
        
        return (count_shared(fwd) + count_shared(rev), balance)
    
    return min(all_solutions, key=solution_score)


def find_bidirectional_decomposition(
    n: int,
    edges: List[Edge],
    max_L: int = 20,
    verbose: bool = False,
    algo: str = "pruned",
) -> Optional[Dict[str, Any]]:
    """
    Find a bidirectional path decomposition where all paths share the same endpoints.
    
    Tries all vertex pairs (s, t) and path lengths L.
    Returns result dict or None.
    """
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
    candidate_lengths = [L for L in range(1, min(m + 1, max_L + 1)) if m % L == 0]
    
    # Prefer longer paths (fewer total paths needed)
    candidate_lengths = sorted(candidate_lengths, reverse=True)
    
    best_result = None
    
    for L in candidate_lengths:
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
            
            # Try to find exact cover
            solution = solve_exact_cover_paths(all_edges_set, valid_paths, valid_edges)
            
            if solution is not None:
                result_paths = [valid_paths[i] for i in solution]
                result_directions = [valid_directions[i] for i in solution]
                
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
                
                # Return first solution found (longest L, first valid pair)
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
                n, edges, max_L=max_L, verbose=verbose, algo=algo
            )
            
            if result is None:
                return {
                    "model": name,
                    "ok": False,
                    "n_vertices": n,
                    "n_edges": len(edges),
                    "vertex_degrees": unique_degrees,
                    "message": "No bidirectional path decomposition found.",
                }
            
            return {
                "model": name,
                "ok": True,
                "n_vertices": n,
                "n_edges": len(edges),
                "vertex_degrees": unique_degrees,
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
        )
        results.append(result)

        if result["ok"]:
            if args.all:
                print(f"  OK: {result['num_solutions']} solutions ({result['num_bidirectional']} bidirectional)", 
                      file=sys.stderr)
            else:
                bidir = "bidirectional (AC)" if result.get('is_bidirectional') else "unidirectional (DC)"
                print(f"  OK: L={result['path_length']}, endpoints={result['endpoint_pair']}, {bidir}", 
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
