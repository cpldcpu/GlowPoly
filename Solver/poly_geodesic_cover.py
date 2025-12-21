#!/usr/bin/env python3
"""
poly_geodesic_cover.py

Test polyhedra graphs (from JSON files in a folder) for shortest-path edge coverage properties.

Two problem variants:

1) mode=edge  (branching allowed):
   Choose pairs (s_i, t_i) with equal distance L such that union of
   E_geo(s_i,t_i) covers all edges.  (Edge-geodetic cover style.)

2) mode=path  (one path per pair):
   Choose pairs and one chosen shortest path per pair such that union of
   chosen paths' edges covers all edges. (Strong-ish shortest path cover.)

Outputs:
- Console summary
- Optional JSON report with chosen pairs and decompositions (paths)

Requires: networkx
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional, Iterable, Set

import networkx as nx


# ----------------------------
# JSON -> Graph parsing
# ----------------------------

def _as_node_id(v: Any, idx_fallback: int) -> Any:
    """Try to extract a stable node id from a vertex record; else fallback to index."""
    if isinstance(v, (int, str)):
        return v
    if isinstance(v, dict):
        for k in ("id", "name", "key", "index"):
            if k in v:
                return v[k]
    return idx_fallback


def load_graph_from_json(path: str, force_directed: Optional[bool] = None) -> Tuple[nx.Graph, str]:
    """
    Accepts a few common shapes:
      - {"name": "...", "vertices":[...], "edges":[[u,v], ...]}
      - {"vertices":[...], "edges":[{"source":u,"target":v}, ...]}
      - {"adjacency": {"0":[1,2], "1":[0], ...}}
    Node ids can be numeric indices or explicit "id"/"name" in vertices.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    name = data.get("name") or os.path.splitext(os.path.basename(path))[0]
    directed = data.get("directed", False)
    if force_directed is not None:
        directed = force_directed

    G = nx.DiGraph() if directed else nx.Graph()

    if "adjacency" in data and isinstance(data["adjacency"], dict):
        # adjacency keys might be strings
        for u, nbrs in data["adjacency"].items():
            G.add_node(u)
            for v in nbrs:
                G.add_edge(u, v)
        return G, name

    vertices = data.get("vertices", [])
    # Build node id list
    node_ids: List[Any] = []
    if isinstance(vertices, list) and len(vertices) > 0:
        for i, v in enumerate(vertices):
            node_ids.append(_as_node_id(v, i))
        for nid in node_ids:
            G.add_node(nid)
    else:
        # No vertices list; nodes will be inferred from edges.
        node_ids = []

    edges = data.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError(f"{path}: 'edges' must be a list")

    def resolve_endpoint(x: Any) -> Any:
        # If vertices existed and x looks like an int index, map it.
        if node_ids and isinstance(x, int) and 0 <= x < len(node_ids):
            return node_ids[x]
        return x

    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            u = resolve_endpoint(e[0])
            v = resolve_endpoint(e[1])
            G.add_edge(u, v)
        elif isinstance(e, dict):
            u = resolve_endpoint(e.get("source", e.get("u")))
            v = resolve_endpoint(e.get("target", e.get("v")))
            if u is None or v is None:
                raise ValueError(f"{path}: edge dict missing endpoints: {e}")
            G.add_edge(u, v)
        else:
            raise ValueError(f"{path}: unsupported edge record: {e}")

    # Ensure connectedness for undirected; for directed we’ll work with weakly connected
    if directed:
        if not nx.is_weakly_connected(G):
            raise ValueError(f"{name}: directed graph is not weakly connected")
    else:
        if not nx.is_connected(G):
            raise ValueError(f"{name}: graph is not connected")

    return G, name


# ----------------------------
# Core geodesic edge coverage
# ----------------------------

def canonical_edge(u: Any, v: Any, directed: bool) -> Tuple[Any, Any]:
    if directed:
        return (u, v)
    return (u, v) if u <= v else (v, u)


def build_edge_index(G: nx.Graph) -> Tuple[Dict[Tuple[Any, Any], int], List[Tuple[Any, Any]]]:
    directed = G.is_directed()
    edges = [canonical_edge(u, v, directed) for (u, v) in G.edges()]
    # Deduplicate in case of weird JSON duplicates
    uniq = list(dict.fromkeys(edges))
    idx = {e: i for i, e in enumerate(uniq)}
    return idx, uniq


def all_pairs_shortest_lengths(G: nx.Graph) -> Dict[Any, Dict[Any, int]]:
    """
    BFS from every node (OK for polyhedra-sized graphs).
    For directed graphs, uses directed distances.
    """
    dist: Dict[Any, Dict[Any, int]] = {}
    for s in G.nodes():
        dist[s] = dict(nx.single_source_shortest_path_length(G, s))
    return dist


def diameter_from_dist(dist: Dict[Any, Dict[Any, int]]) -> int:
    diam = 0
    for s, dmap in dist.items():
        local_max = max(dmap.values())
        diam = max(diam, local_max)
    return diam


def geodesic_edge_mask(
    G: nx.Graph,
    dist: Dict[Any, Dict[Any, int]],
    edge_idx: Dict[Tuple[Any, Any], int],
    s: Any,
    t: Any,
) -> Tuple[int, int]:
    """
    Return (L, mask) where mask marks edges that lie on at least one shortest s-t path.
    Works for undirected or directed graphs.
    """
    if t not in dist[s]:
        return (10**9, 0)
    L = dist[s][t]
    directed = G.is_directed()

    # For directed graphs, need distances *to* t as well (via reverse graph)
    if directed:
        # Precompute reverse distances from t once per call (still cheap for these graphs)
        Grev = G.reverse(copy=False)
        dist_to_t = dict(nx.single_source_shortest_path_length(Grev, t))
    else:
        dist_to_t = dist[t]

    mask = 0
    for (u, v) in G.edges():
        ekey = canonical_edge(u, v, directed)
        ei = edge_idx[ekey]

        if directed:
            # edge (u->v) lies on a shortest s->t path iff d(s,u)+1+d(v,t)=L
            if (u in dist[s]) and (v in dist_to_t) and (dist[s][u] + 1 + dist_to_t[v] == L):
                mask |= (1 << ei)
        else:
            # undirected: either orientation can be used
            du = dist[s].get(u, 10**9)
            dv = dist[s].get(v, 10**9)
            tu = dist_to_t.get(u, 10**9)
            tv = dist_to_t.get(v, 10**9)
            ok = False
            if du + 1 + tv == L:
                ok = True
            elif dv + 1 + tu == L:
                ok = True
            if ok:
                mask |= (1 << ei)

    return L, mask


def max_weight_shortest_path(
    G: nx.Graph,
    dist: Dict[Any, Dict[Any, int]],
    edge_idx: Dict[Tuple[Any, Any], int],
    s: Any,
    t: Any,
    target_mask: int,
    forbidden_opposites: Optional[Set[Tuple[Any, Any]]] = None,
) -> Optional[List[Any]]:
    """
    Find an s->t shortest path that maximizes the number of edges in target_mask.
    Uses DP on the geodesic DAG defined by distance-from-s layers.

    Returns a vertex list path, or None if no path.
    """
    if t not in dist[s]:
        return None
    L = dist[s][t]
    directed = G.is_directed()

    # Distances to t
    if directed:
        Grev = G.reverse(copy=False)
        dist_to_t = dict(nx.single_source_shortest_path_length(Grev, t))
    else:
        dist_to_t = dist[t]

    # Build layered nodes (by dist from s) restricted to interval
    layers: List[List[Any]] = [[] for _ in range(L + 1)]
    for v in G.nodes():
        dv = dist[s].get(v, None)
        if dv is None or dv > L:
            continue
        # Only keep nodes that can still reach t in exactly L-dv steps (needed for geodesic)
        if v in dist_to_t and dv + dist_to_t[v] == L:
            layers[dv].append(v)

    # DP: best score to each node
    NEG = -10**18
    dp: Dict[Any, int] = {v: NEG for v in G.nodes()}
    pred: Dict[Any, Any] = {}
    dp[s] = 0

    # Helper to check if edge can be on some shortest s-t path in the "forward" direction
    def edge_forward(u: Any, v: Any) -> bool:
        if dist[s].get(u, 10**9) + 1 != dist[s].get(v, 10**9):
            return False
        # and must be tight wrt t
        if directed:
            return (u in dist[s]) and (v in dist_to_t) and (dist[s][u] + 1 + dist_to_t[v] == L)
        else:
            return (dist[s].get(u, 10**9) + 1 + dist_to_t.get(v, 10**9) == L)

    for d in range(L):
        for u in layers[d]:
            if dp[u] == NEG:
                continue
            # neighbors / outgoing
            nbrs = G.successors(u) if directed else G.neighbors(u)
            for v in nbrs:
                if not edge_forward(u, v):
                    continue
                if forbidden_opposites is not None and (v, u) in forbidden_opposites:
                    continue
                ekey = canonical_edge(u, v, directed)
                ei = edge_idx[ekey]
                w = 1 if ((target_mask >> ei) & 1) else 0
                val = dp[u] + w
                if val > dp.get(v, NEG):
                    dp[v] = val
                    pred[v] = u

    if dp.get(t, NEG) == NEG:
        return None

    # Reconstruct path
    path = [t]
    cur = t
    while cur != s:
        cur = pred[cur]
        path.append(cur)
    path.reverse()
    return path


def path_edge_mask(G: nx.Graph, edge_idx: Dict[Tuple[Any, Any], int], path: List[Any]) -> int:
    directed = G.is_directed()
    mask = 0
    for a, b in zip(path, path[1:]):
        ekey = canonical_edge(a, b, directed)
        ei = edge_idx[ekey]
        mask |= (1 << ei)
    return mask


def has_opposite_oriented_edges(paths: List[List[Any]]) -> bool:
    """Return True if any edge is used in both directions across the set of paths."""
    seen: Set[Tuple[Any, Any]] = set()
    for p in paths:
        for a, b in zip(p, p[1:]):
            if (b, a) in seen:
                return True
            seen.add((a, b))
    return False


def greedy_pair_cover_edge_mode(
    G: nx.Graph,
    dist: Dict[Any, Dict[Any, int]],
    edge_idx: Dict[Tuple[Any, Any], int],
    L: int,
    max_pairs: int,
    distinct_terminals: bool,
) -> Optional[List[Dict[str, Any]]]:
    """
    Greedy set cover on pairs using geodesic-edge masks.
    Returns list of dicts containing the pair, its full geodesic mask, and the subset newly assigned at selection time.
    """
    nodes = list(G.nodes())
    full_mask = (1 << len(edge_idx)) - 1

    # Candidate pairs at distance L
    cand: List[Tuple[Any, Any, int]] = []
    for i, s in enumerate(nodes):
        for t in nodes[i+1:]:
            if dist[s].get(t, None) == L:
                _, m = geodesic_edge_mask(G, dist, edge_idx, s, t)
                if m != 0:
                    cand.append((s, t, m))

    if not cand:
        return None

    uncovered = full_mask
    used_terminals: Set[Any] = set()
    sol: List[Dict[str, Any]] = []

    for _ in range(max_pairs):
        best = None
        best_gain = 0

        for (s, t, m) in cand:
            if distinct_terminals and (s in used_terminals or t in used_terminals):
                continue
            gain = (m & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best = (s, t, m)

        if best is None or best_gain == 0:
            break

        s, t, m = best
        newly = m & uncovered
        sol.append({
            "s": s,
            "t": t,
            "assigned_mask": newly,
            "geodesic_mask": m,
        })
        uncovered &= ~m
        if distinct_terminals:
            used_terminals.add(s); used_terminals.add(t)
        if uncovered == 0:
            return sol

    return None


def greedy_pair_cover_path_mode(
    G: nx.Graph,
    dist: Dict[Any, Dict[Any, int]],
    edge_idx: Dict[Tuple[Any, Any], int],
    L: int,
    max_pairs: int,
    distinct_terminals: bool,
) -> Optional[List[Dict[str, Any]]]:
    """
    Greedy cover where each chosen pair contributes ONE chosen shortest path (adaptive, max uncovered edges).
    Returns list of dicts with pair + explicit path + assigned edge mask.
    """
    nodes = list(G.nodes())
    full_mask = (1 << len(edge_idx)) - 1
    uncovered = full_mask

    # Candidate pairs at distance L
    cand_pairs: List[Tuple[Any, Any]] = []
    for i, s in enumerate(nodes):
        for t in nodes[i+1:]:
            if dist[s].get(t, None) == L:
                cand_pairs.append((s, t))
    if not cand_pairs:
        return None

    used_terminals: Set[Any] = set()
    sol: List[Dict[str, Any]] = []
    used_oriented_edges: Set[Tuple[Any, Any]] = set()  # Track orientation to forbid opposite reuse

    for _ in range(max_pairs):
        best_item = None
        best_gain = 0

        for (s, t) in cand_pairs:
            if distinct_terminals and (s in used_terminals or t in used_terminals):
                continue
            # Choose best shortest path for current uncovered
            p = max_weight_shortest_path(G, dist, edge_idx, s, t, uncovered, forbidden_opposites=used_oriented_edges)
            if not p:
                continue
            # Reject paths that would reuse an edge in the opposite direction
            conflict = False
            for a, b in zip(p, p[1:]):
                if (b, a) in used_oriented_edges:
                    conflict = True
                    break
            if conflict:
                continue
            pm = path_edge_mask(G, edge_idx, p)
            gain = (pm & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_item = (s, t, p, pm)

        if best_item is None or best_gain == 0:
            break

        s, t, p, pm = best_item
        newly = pm & uncovered
        sol.append({
            "s": s,
            "t": t,
            "path": p,
            "assigned_mask": newly,
            "path_mask": pm,
            "path_edges_count": max(len(p) - 1, 0),
        })
        # Record oriented usage to prevent opposite-direction reuse later
        for a, b in zip(p, p[1:]):
            used_oriented_edges.add((a, b))
        uncovered &= ~pm
        if distinct_terminals:
            used_terminals.add(s); used_terminals.add(t)
        if uncovered == 0:
            return sol

    return None


def greedy_path_decomposition_for_assigned_edges(
    G: nx.Graph,
    dist: Dict[Any, Dict[Any, int]],
    edge_idx: Dict[Tuple[Any, Any], int],
    s: Any,
    t: Any,
    assigned_mask: int,
    max_paths: int = 200,
    used_oriented_edges: Optional[Set[Tuple[Any, Any]]] = None,
) -> List[List[Any]]:
    """
    Given one pair (s,t) and a set of edges to cover, repeatedly pick a shortest path that
    covers as many remaining assigned edges as possible (DP in geodesic DAG).
    """
    paths: List[List[Any]] = []
    remaining = assigned_mask
    used_oriented_edges = used_oriented_edges if used_oriented_edges is not None else set()
    for _ in range(max_paths):
        if remaining == 0:
            break
        p = max_weight_shortest_path(G, dist, edge_idx, s, t, remaining, forbidden_opposites=used_oriented_edges)
        if not p:
            break
        conflict = False
        for a, b in zip(p, p[1:]):
            if (b, a) in used_oriented_edges:
                conflict = True
                break
        if conflict:
            break
        pm = path_edge_mask(G, edge_idx, p)
        gain = (pm & remaining).bit_count()
        if gain == 0:
            break
        paths.append(p)
        for a, b in zip(p, p[1:]):
            used_oriented_edges.add((a, b))
        remaining &= ~pm
    return paths


# ----------------------------
# Reporting
# ----------------------------

@dataclass
class PolyResult:
    name: str
    nV: int
    nE: int
    directed: bool
    diameter: int
    pair_edge_counts: Optional[List[int]] = None

    # Best single-pair (edge mode) if exists
    single_pair_L: Optional[int] = None
    single_pair: Optional[Tuple[Any, Any]] = None

    # Best multi-pair solution found
    best_L: Optional[int] = None
    best_k: Optional[int] = None
    solution: Optional[Any] = None  # list with details


def analyze_one(
    G: nx.Graph,
    name: str,
    mode: str,
    max_pairs: int,
    distinct_terminals: bool,
    want_paths: bool,
    objective: str,
    max_L: Optional[int],
) -> PolyResult:
    edge_idx, _ = build_edge_index(G)
    dist = all_pairs_shortest_lengths(G)
    diam = diameter_from_dist(dist)

    res = PolyResult(
        name=name,
        nV=G.number_of_nodes(),
        nE=G.number_of_edges(),
        directed=G.is_directed(),
        diameter=diam,
    )

    full_mask = (1 << len(edge_idx)) - 1
    nodes = list(G.nodes())

    # First: detect a single-pair edge-mode solution (this is your cube property)
    best_single = None
    best_single_pair = None
    for i, s in enumerate(nodes):
        for t in nodes[i+1:]:
            if t not in dist[s]:
                continue
            L, m = geodesic_edge_mask(G, dist, edge_idx, s, t)
            if m == full_mask:
                if best_single is None or L < best_single:
                    best_single = L
                    best_single_pair = (s, t)
    res.single_pair_L = best_single
    res.single_pair = best_single_pair

    L_limit = diam if max_L is None else min(diam, max_L)

    candidates: List[Dict[str, Any]] = []

    # Now: find best cover under requested mode (collect all valid, then choose objective)
    for L in range(1, L_limit + 1):
        if mode == "edge":
            sol = greedy_pair_cover_edge_mode(G, dist, edge_idx, L, max_pairs, distinct_terminals)
            if sol is None:
                continue
            # Verify coverage
            covered = 0
            for item in sol:
                covered |= item["assigned_mask"]
            if covered != full_mask:
                # assigned only tracks newly covered at selection time; still union should equal full if success
                pass
            res.best_L = L
            res.best_k = len(sol)

            detailed = []
            used_oriented_edges: Set[Tuple[Any, Any]] = set() if want_paths else set()
            for item in sol:
                geodesic_count = item["geodesic_mask"].bit_count()
                assigned = item["assigned_mask"]
                entry = {
                    "s": item["s"],
                    "t": item["t"],
                    "assigned_edges_count": assigned.bit_count(),
                    "geodesic_edges_count": geodesic_count,
                }
                if want_paths:
                    entry["paths"] = greedy_path_decomposition_for_assigned_edges(
                        G, dist, edge_idx, item["s"], item["t"], assigned, used_oriented_edges=used_oriented_edges
                    )
                detailed.append(entry)

            if want_paths:
                all_paths: List[List[Any]] = []
                for d in detailed:
                    all_paths.extend(d.get("paths", []))
                if has_opposite_oriented_edges(all_paths):
                    # Invalid: overlapping edges in opposite directions; keep searching
                    continue
                # Require that each pair with assigned edges has at least one covering path
                invalid = False
                for d in detailed:
                    if d["assigned_edges_count"] > 0 and not d.get("paths"):
                        invalid = True
                        break
                if invalid:
                    continue

            candidates.append({
                "L": L,
                "k": len(sol),
                "solution": detailed,
                "pair_edge_counts": [d["geodesic_edges_count"] for d in detailed],
            })

        elif mode == "path":
            sol = greedy_pair_cover_path_mode(G, dist, edge_idx, L, max_pairs, distinct_terminals)
            if sol is None:
                continue
            # Verify coverage
            union = 0
            for item in sol:
                union |= item["assigned_mask"]
            # assigned_mask is newly covered; union==full means success
            if union != full_mask:
                continue
            # Reject if opposite directions used on same edge
            all_paths = [item["path"] for item in sol if item.get("path")]
            if has_opposite_oriented_edges(all_paths):
                # Invalid due to opposite directions on same edge; try higher L
                continue

            # Convert masks to counts only (keep paths)
            detailed = []
            for item in sol:
                path_edges_count = item.get("path_edges_count", max(len(item["path"]) - 1, 0))
                detailed.append({
                    "s": item["s"],
                    "t": item["t"],
                    "path": item["path"],
                    "assigned_edges_count": int(item["assigned_mask"].bit_count()),
                    "path_edges_count": path_edges_count,
                })
            candidates.append({
                "L": L,
                "k": len(sol),
                "solution": detailed,
                "pair_edge_counts": [d["path_edges_count"] for d in detailed],
            })
        else:
            raise ValueError("mode must be 'edge' or 'path'")

    if candidates:
        if objective == "min_pairs":
            best = min(candidates, key=lambda c: (c["k"], c["L"]))
        else:  # min_L
            best = min(candidates, key=lambda c: (c["L"], c["k"]))
        res.best_L = best["L"]
        res.best_k = best["k"]
        res.solution = best["solution"]
        res.pair_edge_counts = best["pair_edge_counts"]

    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder containing *.json polyhedra graphs")
    ap.add_argument("--mode", choices=["edge", "path"], default="edge",
                    help="edge: branching geodesic-edge cover; path: one chosen shortest path per pair")
    ap.add_argument("--max_pairs", type=int, default=6, help="Max number of pairs to use in a cover")
    ap.add_argument("--distinct_terminals", action="store_true",
                    help="Require all endpoints across pairs to be distinct (2k distinct vertices)")
    ap.add_argument("--directed", action="store_true", help="Force treating graphs as directed")
    ap.add_argument("--undirected", action="store_true", help="Force treating graphs as undirected")
    ap.add_argument("--paths", action="store_true",
                    help="In edge mode: also output a greedy set of explicit shortest paths per pair to cover assigned edges")
    ap.add_argument("--objective", choices=["min_L", "min_pairs"], default="min_L",
                    help="min_L: prefer smallest path length L; min_pairs: prefer smallest number of pairs (tie-break on L)")
    ap.add_argument("--max_L", type=int, default=None, help="Optional upper bound on path length L to search (default: diameter)")
    ap.add_argument("--out_json", default="geodesic_cover_results.json", help="Write results JSON here")
    args = ap.parse_args()

    if args.directed and args.undirected:
        raise ValueError("Choose at most one of --directed / --undirected")

    force_directed = True if args.directed else (False if args.undirected else None)

    results: List[PolyResult] = []
    files = [f for f in os.listdir(args.folder) if f.lower().endswith(".json")]
    files.sort()

    for fn in files:
        path = os.path.join(args.folder, fn)
        try:
            G, name = load_graph_from_json(path, force_directed=force_directed)
            r = analyze_one(
                G, name,
                mode=args.mode,
                max_pairs=args.max_pairs,
                distinct_terminals=args.distinct_terminals,
                want_paths=args.paths,
                objective=args.objective,
                max_L=args.max_L,
            )
            results.append(r)
        except Exception as e:
            print(f"[FAIL] {fn}: {e}")

    # Console summary
    print(f"{'name':30s}  V   E   diam  singlePair(L)  best(L,k)  pairEdges")
    print("-" * 100)
    for r in results:
        sp = f"{r.single_pair_L}" if r.single_pair_L is not None else "-"
        bk = f"{r.best_L},{r.best_k}" if r.best_L is not None else "-,-"
        pc = "/".join(str(x) for x in r.pair_edge_counts) if r.pair_edge_counts else "-"
        print(f"{r.name[:30]:30s}  {r.nV:3d} {r.nE:3d}  {r.diameter:4d}     {sp:>6s}       {bk:>8s}    {pc}")

    # JSON output
    out = [asdict(r) for r in results]
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nWrote: {args.out_json}")


if __name__ == "__main__":
    main()
