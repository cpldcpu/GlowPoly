#!/usr/bin/env python3
"""
Equal Cycle Decomposition Solver (length = 2 * diameter)

Finds a partition of all edges into edge-disjoint cycles where:
  1. All cycles have the SAME length (2 * graph diameter)
  2. All cycles are CHORDLESS (no shortcuts)
  3. Opposite vertex pairs on each cycle have distance == diameter
  4. Output includes DIRECTED edges (order follows cycle vertex sequence)

Usage:
  python equal_cycle_decomp.py model.json
  python equal_cycle_decomp.py models_folder/ --out results/
"""

import argparse
import json
import os
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

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


def all_pairs_shortest_paths(adj: Dict[int, Set[int]]) -> Optional[List[List[int]]]:
    """Return all-pairs shortest-path distances or None if disconnected."""
    n = len(adj)
    dist_matrix: List[List[int]] = []
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if any(d == -1 for d in dist):
            return None
        dist_matrix.append(dist)
    return dist_matrix


def graph_diameter_from_dist(dist_matrix: Optional[List[List[int]]]) -> Optional[int]:
    if dist_matrix is None:
        return None
    if not dist_matrix:
        return 0
    return max(max(row) for row in dist_matrix)


def enumerate_cycles(adj: Dict[int, Set[int]], length: int, max_cycles: int = 50000) -> List[List[int]]:
    """
    Enumerate simple cycles of exactly 'length' vertices.
    Stops early if max_cycles is reached to prevent exponential blowup.
    """
    n = len(adj)
    found: Set[Tuple[int, ...]] = set()

    for start in range(n):
        if len(found) >= max_cycles:
            break
            
        # DFS: (current_vertex, path, visited)
        stack: List[Tuple[int, List[int], Set[int]]] = [(start, [start], {start})]

        while stack:
            if len(found) >= max_cycles:
                break
                
            v, path, visited = stack.pop()

            if len(path) == length:
                # Check if we can close the cycle
                if start in adj[v]:
                    # Canonicalize: smallest rotation of min(forward, reverse)
                    cyc = tuple(path)
                    rev = tuple(reversed(path))
                    rotations = []
                    for i in range(length):
                        rotations.append(cyc[i:] + cyc[:i])
                        rotations.append(rev[i:] + rev[:i])
                    canon = min(rotations)
                    found.add(canon)
                continue

            if len(path) >= length:
                continue

            for u in adj[v]:
                if u == start and len(path) >= 3:
                    # Can close early - but we want exact length
                    continue
                if u < start:
                    # Ensure start is minimum vertex (avoid duplicates)
                    continue
                if u in visited:
                    continue
                stack.append((u, path + [u], visited | {u}))

    return [list(c) for c in sorted(found)]


def edges_of_cycle(cycle: List[int]) -> List[Edge]:
    """Get undirected edges (min, max) from cycle."""
    k = len(cycle)
    return [(min(cycle[i], cycle[(i+1) % k]), max(cycle[i], cycle[(i+1) % k])) for i in range(k)]


def directed_edges_of_cycle(cycle: List[int]) -> List[Tuple[int, int]]:
    """Get directed edges following vertex order."""
    k = len(cycle)
    return [(cycle[i], cycle[(i+1) % k]) for i in range(k)]


def has_chord(cycle: List[int], adj: Dict[int, Set[int]]) -> bool:
    """
    Check if cycle has a chord (edge between non-adjacent vertices).
    Returns True if a chord exists.
    """
    k = len(cycle)
    if k < 4:
        return False  # Triangles can't have chords
    
    for i in range(k):
        vi = cycle[i]
        # Check vertices that are not adjacent in the cycle
        for offset in range(2, k - 1):
            j = (i + offset) % k
            vj = cycle[j]
            if vj in adj[vi]:
                return True  # Found a chord
    return False


def cycle_opposite_pairs_ok(cycle: List[int], dist_matrix: List[List[int]], diameter: int) -> bool:
    """Opposite vertices on the cycle must be at graph distance == diameter."""
    k = len(cycle)
    if k % 2 != 0:
        return False
    half = k // 2
    if half != diameter:
        return False
    for i in range(half):
        u = cycle[i]
        v = cycle[i + half]
        if dist_matrix[u][v] != diameter:
            return False
    return True


def count_paths_of_length(adj: Dict[int, Set[int]], start: int, end: int, length: int) -> int:
    """
    Count all simple paths of exactly 'length' edges from start to end.
    """
    if length == 0:
        return 1 if start == end else 0
    if length < 0:
        return 0
    
    count = 0
    # DFS with path tracking
    stack = [(start, {start}, 0)]  # (current, visited, depth)
    
    while stack:
        curr, visited, depth = stack.pop()
        
        if depth == length:
            if curr == end:
                count += 1
            continue
        
        for neighbor in adj[curr]:
            if neighbor not in visited:
                stack.append((neighbor, visited | {neighbor}, depth + 1))
    
    return count


def has_branch(cycle: List[int], adj: Dict[int, Set[int]]) -> bool:
    """
    Check if cycle has a branch point.
    
    A branch exists if any intermediate vertex on a half-path has an
    alternative route (of the same remaining length) to the endpoint
    that doesn't follow the cycle's path.
    
    For cycle [0, 1, 2, 5, 4, 3] with endpoints (0, 5):
    - path1: 0 → 1 → 2 → 5
    - At vertex 1, if there's another path of length 2 to 5 (like 1 → 4 → 5),
      that's a branch point.
    
    Returns True if a branch exists (cycle is NOT valid for the constraint).
    """
    k = len(cycle)
    if k < 4 or k % 2 != 0:
        return False
    
    half = k // 2
    
    # Get the two half-paths
    # path1: cycle[0] -> cycle[half] (vertices 0, 1, ..., half)
    # path2: cycle[half] -> cycle[0] (vertices half, half+1, ..., k-1, 0)
    s = cycle[0]
    t = cycle[half]
    
    path1 = cycle[:half + 1]  # 0 to half inclusive
    path2 = cycle[half:] + [cycle[0]]  # half to end, then wrap to 0
    
    # Check path1 for branches: at each intermediate vertex,
    # is there an alternative route to the endpoint?
    for i in range(1, len(path1) - 1):  # Skip start and end
        v = path1[i]
        remaining_length = half - i  # edges remaining to endpoint
        
        # Count paths from v to t of length remaining_length,
        # excluding the vertices already used in path1 up to v
        used_before = set(path1[:i])  # vertices before v in path1
        
        # Check neighbors of v (excluding the path's next vertex)
        next_on_path = path1[i + 1]
        for neighbor in adj[v]:
            if neighbor == next_on_path:
                continue  # This is the path we're on
            if neighbor in used_before:
                continue  # Can't go back
            
            # Is there a path of (remaining_length - 1) from neighbor to t?
            # that doesn't use vertices before v
            path_count = count_paths_of_length_avoiding(
                adj, neighbor, t, remaining_length - 1, used_before | {v}
            )
            if path_count > 0:
                return True  # Found a branch!
    
    # Check path2 for branches similarly
    for i in range(1, len(path2) - 1):  # Skip start (t) and end (s)
        v = path2[i]
        remaining_length = half - i
        
        used_before = set(path2[:i])
        next_on_path = path2[i + 1]
        
        for neighbor in adj[v]:
            if neighbor == next_on_path:
                continue
            if neighbor in used_before:
                continue
            
            path_count = count_paths_of_length_avoiding(
                adj, neighbor, s, remaining_length - 1, used_before | {v}
            )
            if path_count > 0:
                return True
    
    return False


def count_paths_of_length_avoiding(
    adj: Dict[int, Set[int]], start: int, end: int, length: int, avoid: Set[int]
) -> int:
    """
    Count simple paths of exactly 'length' edges from start to end,
    avoiding the vertices in 'avoid'.
    """
    if start in avoid or end in avoid:
        return 0
    if length == 0:
        return 1 if start == end else 0
    if length < 0:
        return 0
    
    count = 0
    stack = [(start, avoid | {start}, 0)]
    
    while stack:
        curr, visited, depth = stack.pop()
        
        if depth == length:
            if curr == end:
                count += 1
            continue
        
        for neighbor in adj[curr]:
            if neighbor not in visited:
                stack.append((neighbor, visited | {neighbor}, depth + 1))
    
    return count


def get_opposite_endpoints(cycle: List[int]) -> List[Tuple[int, int]]:
    """
    For an even-length cycle, find all possible (start, end) pairs
    where start and end are opposite vertices (L/2 apart).
    Returns list of (start, end) tuples, canonicalized as (min, max).
    """
    k = len(cycle)
    if k % 2 != 0:
        return []  # Odd cycles don't have opposite vertices
    
    half = k // 2
    pairs = []
    for i in range(half):
        s = cycle[i]
        t = cycle[i + half]
        pairs.append((min(s, t), max(s, t)))
    return pairs


def orient_cycle_for_endpoints(cycle: List[int], start: int, end: int) -> List[int]:
    """
    Rotate and possibly reverse cycle so it starts at 'start' and 
    the opposite vertex is 'end'. Returns the oriented cycle as two paths:
    start -> ... -> end (first half) and end -> ... -> start (second half).
    """
    k = len(cycle)
    half = k // 2
    
    # Find start position
    try:
        start_idx = cycle.index(start)
    except ValueError:
        return cycle  # start not in cycle
    
    # Rotate so start is first
    rotated = cycle[start_idx:] + cycle[:start_idx]
    
    # Check if end is at position half
    if rotated[half] == end:
        return rotated
    
    # Try reverse
    rev = list(reversed(cycle))
    try:
        start_idx = rev.index(start)
    except ValueError:
        return cycle
    rotated = rev[start_idx:] + rev[:start_idx]
    
    if rotated[half] == end:
        return rotated
    
    return cycle  # Fallback


def minimize_endpoints(cycles: List[List[int]]) -> Tuple[List[Tuple[int, int]], Dict[int, List[int]]]:
    """
    Given a list of cycles, choose endpoints for each to minimize
    the total number of unique endpoints.
    
    Uses exhaustive search for small cycle counts, greedy for larger.
    
    Returns:
        - List of (start, end) pairs for each cycle
        - Dict mapping each unique endpoint to list of cycle indices using it
    """
    from itertools import product
    
    n_cycles = len(cycles)
    if n_cycles == 0:
        return [], {}
    
    # Get all possible endpoint pairs for each cycle
    all_pairs = [get_opposite_endpoints(c) for c in cycles]
    
    # Handle cycles with no valid pairs (odd length - shouldn't happen)
    for i, pairs in enumerate(all_pairs):
        if not pairs:
            k = len(cycles[i])
            s, t = cycles[i][0], cycles[i][k // 2] if k > 1 else cycles[i][0]
            all_pairs[i] = [(min(s, t), max(s, t))]
    
    # For small number of cycles, do exhaustive search
    # Product of choices is manageable for up to ~10 cycles with 3-6 options each
    total_combinations = 1
    for pairs in all_pairs:
        total_combinations *= len(pairs)
    
    if total_combinations <= 100000:  # Exhaustive search
        best_assignment = None
        best_unique_count = float('inf')
        
        for combo in product(*all_pairs):
            unique = set()
            for s, t in combo:
                unique.add(s)
                unique.add(t)
            if len(unique) < best_unique_count:
                best_unique_count = len(unique)
                best_assignment = list(combo)
        
        assigned = best_assignment
    else:
        # Fall back to greedy for very large search spaces
        assigned: List[Tuple[int, int]] = []  # type: ignore
        endpoint_count: Dict[int, int] = {}
        
        order = sorted(range(n_cycles), key=lambda i: len(all_pairs[i]))
        temp_assigned = [None] * n_cycles
        
        for ci in order:
            pairs = all_pairs[ci]
            best_pair = None
            best_score = -1
            for s, t in pairs:
                score = endpoint_count.get(s, 0) + endpoint_count.get(t, 0)
                if score > best_score:
                    best_score = score
                    best_pair = (s, t)
            
            if best_pair is None:
                best_pair = pairs[0]
            
            temp_assigned[ci] = best_pair
            s, t = best_pair
            endpoint_count[s] = endpoint_count.get(s, 0) + 1
            endpoint_count[t] = endpoint_count.get(t, 0) + 1
        
        assigned = temp_assigned  # type: ignore
    
    # Build endpoint -> cycles mapping
    endpoint_to_cycles: Dict[int, List[int]] = {}
    for ci, (s, t) in enumerate(assigned):
        endpoint_to_cycles.setdefault(s, []).append(ci)
        endpoint_to_cycles.setdefault(t, []).append(ci)
    
    return assigned, endpoint_to_cycles


def solve_exact_cover(
    all_edges: Set[Edge],
    cycles: List[List[int]],
    cycle_edges: List[Set[Edge]],
) -> Optional[List[int]]:
    """
    Find a subset of cycles that covers all edges exactly once.
    Returns list of cycle indices, or None if impossible.
    """
    # Build edge -> candidate cycles mapping
    edge_to_cycles: Dict[Edge, List[int]] = {e: [] for e in all_edges}
    for ci, es in enumerate(cycle_edges):
        for e in es:
            if e in edge_to_cycles:
                edge_to_cycles[e].append(ci)

    # Quick fail: any edge with no covering cycle
    for e, cands in edge_to_cycles.items():
        if not cands:
            return None

    # Backtracking with most-constrained-edge heuristic
    uncovered = set(all_edges)
    chosen: List[int] = []
    cycle_edge_sets = cycle_edges  # already sets

    def pick_edge() -> Edge:
        best_e = None
        best_count = float('inf')
        for e in uncovered:
            count = sum(1 for ci in edge_to_cycles[e] if cycle_edge_sets[ci].issubset(uncovered))
            if count < best_count:
                best_count = count
                best_e = e
                if count <= 1:
                    break
        return best_e  # type: ignore

    def backtrack() -> bool:
        if not uncovered:
            return True

        e = pick_edge()
        if e is None:
            return False

        for ci in edge_to_cycles[e]:
            ces = cycle_edge_sets[ci]
            if not ces.issubset(uncovered):
                continue

            # Choose this cycle
            chosen.append(ci)
            uncovered.difference_update(ces)

            if backtrack():
                return True

            # Undo
            uncovered.update(ces)
            chosen.pop()

        return False

    if backtrack():
        return chosen
    return None


def find_decomposition(
    n: int,
    edges: List[Edge],
    prefer_longer: bool = True,
    max_L: int = 20,
    require_unique: bool = False,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Find cycle decomposition with equal cycle length 2*diameter.
    
    Args:
        require_unique: If True, only accept geodesically unique cycles
                        (no alternative paths of same length between endpoints)
    
    Returns result dict or None.
    """
    m = len(edges)
    adj = build_adj(n, edges)

    # Check Eulerian condition: all degrees must be even
    for v, nbrs in adj.items():
        if len(nbrs) % 2 != 0:
            return None  # Not Eulerian

    dist_matrix = all_pairs_shortest_paths(adj)
    diameter = graph_diameter_from_dist(dist_matrix)
    if diameter is None:
        return None
    L = 2 * diameter
    if L < 4:
        return None
    if m % L != 0:
        return None
    if max_L is not None and L > max_L:
        return None

    candidates = [L]

    all_edges_set = set(edges)

    for L in candidates:
        if verbose:
            print(f"    Trying L={L}...", file=sys.stderr, end=" ", flush=True)
        cycles = enumerate_cycles(adj, L)
        if not cycles:
            if verbose:
                print(f"0 cycles", file=sys.stderr)
            continue

        # Build edge sets for each cycle
        cycle_edges = [set(edges_of_cycle(c)) for c in cycles]

        # Filter: only keep cycles that are:
        # 1. Using valid edges
        # 2. Chordless (no shortcuts via non-cycle edges)
        # 3. Opposite vertex pairs have distance == diameter
        # 4. Optionally: No branching (no alternative paths from intermediate vertices)
        valid = []
        valid_edges = []
        for c, es in zip(cycles, cycle_edges):
            if not es.issubset(all_edges_set):
                continue
            if has_chord(c, adj):
                continue
            if dist_matrix is None or not cycle_opposite_pairs_ok(c, dist_matrix, diameter):
                continue
            if require_unique and has_branch(c, adj):
                continue
            valid.append(c)
            valid_edges.append(es)

        if not valid:
            if verbose:
                print(f"0 valid", file=sys.stderr)
            continue
        
        if verbose:
            print(f"{len(valid)} valid", file=sys.stderr)

        # Solve exact cover
        solution = solve_exact_cover(all_edges_set, valid, valid_edges)

        if solution is not None:
            result_cycles = [valid[i] for i in solution]
            
            # Optimize endpoints
            endpoints, endpoint_map = minimize_endpoints(result_cycles)
            
            # Orient cycles based on chosen endpoints
            oriented_cycles = []
            for ci, cyc in enumerate(result_cycles):
                s, t = endpoints[ci]
                oriented = orient_cycle_for_endpoints(cyc, s, t)
                half = len(oriented) // 2
                # Split into two paths: start->end and end->start
                path1 = oriented[:half + 1]  # start to end
                path2 = oriented[half:] + [oriented[0]]  # end to start (complete loop)
                oriented_cycles.append({
                    "vertices": oriented,
                    "start": s,
                    "end": t,
                    "path1": path1,
                    "path2": path2,
                    "directed_edges": directed_edges_of_cycle(oriented),
                })
            
            unique_endpoints = set()
            for s, t in endpoints:
                unique_endpoints.add(s)
                unique_endpoints.add(t)
            
            return {
                "cycle_length": L,
                "num_cycles": len(result_cycles),
                "num_unique_endpoints": len(unique_endpoints),
                "unique_endpoints": sorted(unique_endpoints),
                "diameter": diameter,
                "cycles": oriented_cycles,
            }

    return None


def compute_cycle_weights(
    cycles: List[Dict[str, Any]], 
    all_edges: List[Edge]
) -> Dict[str, Any]:
    """
    Compute optimal cycle weights so all edges have equal total flow.
    
    Uses least-squares optimization: A @ w = 1 where A is the edge-cycle 
    incidence matrix (A[i,j] = 1 if edge i is in cycle j, else 0).
    
    Args:
        cycles: List of cycle dicts with "vertices" field
        all_edges: List of all graph edges
    
    Returns:
        Dict with weights, edge_flows, uniformity score, etc.
    """
    import numpy as np
    
    edges_list = sorted(all_edges)
    n_edges = len(edges_list)
    n_cycles = len(cycles)
    
    if n_edges == 0 or n_cycles == 0:
        return {
            "weights": [],
            "edge_flows": {},
            "uniformity": 0.0,
            "max_flow": 0.0,
            "min_flow": 0.0,
        }
    
    # Build incidence matrix A[edge_idx][cycle_idx] = 1 if edge in cycle
    A = np.zeros((n_edges, n_cycles))
    edge_to_idx = {e: i for i, e in enumerate(edges_list)}
    
    for j, cyc in enumerate(cycles):
        cyc_edges = edges_of_cycle(cyc["vertices"])
        for e in cyc_edges:
            if e in edge_to_idx:
                A[edge_to_idx[e], j] = 1.0
    
    # Solve A @ w = 1 (target: uniform flow of 1 per edge)
    target = np.ones(n_edges)
    w, residuals, rank, s = np.linalg.lstsq(A, target, rcond=None)
    w = np.clip(w, 0, None)  # Non-negative weights
    
    # Normalize weights so max is 1.0 (like geodesic covers)
    w_max = w.max()
    if w_max > 0:
        w_normalized = w / w_max
    else:
        w_normalized = w
    
    # Compute resulting edge flows with normalized weights
    edge_flows = A @ w
    
    # Uniformity: 1 - coefficient of variation (1.0 = perfect)
    if edge_flows.mean() > 0:
        uniformity = 1.0 - (edge_flows.std() / edge_flows.mean())
    else:
        uniformity = 0.0
    
    return {
        "weights": w_normalized.tolist(),
        "weights_raw": w.tolist(),
        "edge_flows": {f"{e[0]}-{e[1]}": float(f) for e, f in zip(edges_list, edge_flows)},
        "uniformity": round(uniformity, 4),
        "max_flow": float(edge_flows.max()),
        "min_flow": float(edge_flows.min()),
    }


def process_model(path: str, max_L: int = 20, require_unique: bool = False, verbose: bool = False) -> Dict[str, Any]:
    """Process a single model file."""
    name = os.path.basename(path)
    try:
        n, edges = load_model(path)
        
        # Calculate vertex degrees
        adj = build_adj(n, edges)
        degrees = [len(adj[v]) for v in range(n) if v in adj]
        unique_degrees = sorted(set(degrees))
        
        result = find_decomposition(n, edges, max_L=max_L, require_unique=require_unique, verbose=verbose)

        if result is None:
            return {
                "model": name,
                "ok": False,
                "n_vertices": n,
                "n_edges": len(edges),
                "vertex_degrees": unique_degrees,
                "message": "No equal even-length cycle decomposition found.",
            }

        # Compute optimal cycle weights for uniform edge brightness
        weight_info = compute_cycle_weights(result["cycles"], edges)

        return {
            "model": name,
            "ok": True,
            "n_vertices": n,
            "n_edges": len(edges),
            "vertex_degrees": unique_degrees,
            **result,
            "cycle_weights": weight_info["weights"],
            "uniformity": weight_info["uniformity"],
        }

    except Exception as e:
        return {
            "model": name,
            "ok": False,
            "error": str(e),
        }


def main():
    ap = argparse.ArgumentParser(description="Equal cycle decomposition solver (length = 2 * diameter)")
    ap.add_argument("input", help="Model JSON file or folder of models")
    ap.add_argument("--out", "-o", default=None, help="Output folder (default: print to stdout)")
    ap.add_argument(
        "--max-L",
        type=int,
        default=20,
        help="Safety cap: skip if 2*diameter exceeds this value (default: 20)",
    )
    ap.add_argument("--unique", action="store_true", help="Only accept geodesically unique cycles")
    ap.add_argument("--verbose", "-v", action="store_true", help="Show progress for each cycle length")
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
        result = process_model(path, max_L=args.max_L, require_unique=args.unique, verbose=args.verbose)
        results.append(result)

        if result["ok"]:
            ep = result.get('num_unique_endpoints', '?')
            print(f"  OK: L={result['cycle_length']}, {result['num_cycles']} cycles, {ep} endpoints", file=sys.stderr)
        else:
            print(f"  FAIL: {result.get('message', result.get('error', 'unknown'))}", file=sys.stderr)

    # Output
    if args.out:
        import csv
        os.makedirs(args.out, exist_ok=True)
        
        # Write combined results JSON (like geodesic_cover_results.json)
        results_path = os.path.join(args.out, "cycle_decomp_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        # Write summary CSV
        summary_path = os.path.join(args.out, "summary.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "ok", "n_vertices", "n_edges", "vertex_degrees", "cycle_length", "num_cycles", "num_endpoints", "note"])
            for r in results:
                # Format vertex_degrees as string like "3,4" or "4"
                vd = r.get("vertex_degrees", [])
                vd_str = ",".join(str(d) for d in vd) if vd else ""
                writer.writerow([
                    r.get("model", ""),
                    r.get("ok", False),
                    r.get("n_vertices", ""),
                    r.get("n_edges", ""),
                    vd_str,
                    r.get("cycle_length", ""),
                    r.get("num_cycles", ""),
                    r.get("num_unique_endpoints", ""),
                    r.get("message", r.get("error", "")),
                ])
        
        print(f"\nWrote {len(results)} result(s) to {args.out}/", file=sys.stderr)
        print(f"Results: {results_path}", file=sys.stderr)
        print(f"Summary: {summary_path}", file=sys.stderr)
    else:
        # Single file: print result
        if len(results) == 1:
            print(json.dumps(results[0], indent=2))
        else:
            print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

