"""
Polyhedron LED Path Optimization Solver

This module provides constraint-based optimization for LED path planning on polyhedra.

Includes a special two-feedpoint solver for finding vertex pairs where all shortest
paths between them cover all edges (edge-geodetic pairs).

CORE ARCHITECTURE:
- Generic constraint-based solvers (exhaustive_orientations_with_constraints, sampled_orientations_with_constraints)
- Composable constraint functions (constraint_dc_only, constraint_equal_current, etc.)
- Unified optimization objective: minimize distinct endpoints, then minimize paths

OPTIMIZATION OBJECTIVES:
1) Primary: minimize the number of DISTINCT endpoint vertices used by the chosen paths
2) Constraint: all chosen paths must have a common fixed length L (provided by the user)  
3) Tiebreaker: among those, minimize the NUMBER OF PATHS
4) Secondary tiebreaker: smaller max per-vertex endpoint usage; then lexicographic

CONSTRAINT TYPES:
- DC only: vertices are either pure Anode or pure Cathode (no alternating)
- Equal current: all edges carry equal current flow
- Sneak free: no shorter DC paths exist than solver paths
- Alternating only: all vertices are alternating (both Anode and Cathode)
- Bipolar only: can be implemented without tristate (Z) disconnections

USAGE:
Use exhaustive_orientations_with_constraints() or sampled_orientations_with_constraints()
with a list of constraint functions for flexible, maintainable optimization.
"""
from collections import defaultdict, deque, Counter
import random
from polyhedra import PolyhedronGenerators

# -------------------- Graph + geodesics --------------------

class DiGraph:
    def __init__(self, V, edges):
        self.V = list(V)
        self.edges = list(edges)
        self.adj = defaultdict(list)
        for u,v in self.edges:
            self.adj[u].append(v)
        self.edge_index = {e:i for i,e in enumerate(self.edges)}
    def out_neighbors(self, u):
        return self.adj[u]

def bfs_distances_dir(G, s):
    dist = {s:0}
    dq = deque([s])
    while dq:
        u = dq.popleft()
        for v in G.out_neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist

def enumerate_shortest_paths_dir(G, s, t):
    if s == t: return []
    dist = bfs_distances_dir(G, s)
    if t not in dist: return []
    d = dist[t]
    dag = defaultdict(list)
    for (u,v) in G.edges:
        if (u in dist) and (v in dist) and (dist[v] == dist[u] + 1):
            dag[u].append(v)
    # Only enumerate shortest paths (length d)
    paths = []
    stack = [(s, [s])]
    while stack:
        u, P = stack.pop()
        if u == t:
            if len(P)-1 == d:
                paths.append(P)
            continue
        for v in dag[u]:
            stack.append((v, P+[v]))
    return paths

def all_geodesic_paths_dir(G):
    paths, masks, lens = [], [], []
    for s in G.V:
        dist = bfs_distances_dir(G, s)
        for t in dist.keys():
            if t == s: continue
            for P in enumerate_shortest_paths_dir(G, s, t):
                mask = 0
                for u,v in zip(P,P[1:]):
                    mask |= (1 << G.edge_index[(u,v)])
                paths.append(P)
                masks.append(mask)
                lens.append(len(P)-1)
    return paths, masks, lens

# -------------------- Exact cover --------------------

class ExactCoverMinRows:
    def __init__(self, num_cols, row_masks):
        self.N = num_cols
        self.rows = row_masks
        self.col_to_rows = defaultdict(list)
        for r_idx, mask in enumerate(self.rows):
            m = mask
            while m:
                c = (m & -m).bit_length()-1
                self.col_to_rows[c].append(r_idx)
                m &= m-1
        self.best_solution = None
        self.best_len = float('inf')
        self.search_calls = 0
        self.max_search_calls = 1000000  # Limit to prevent infinite recursion
    def _choose_column(self, remaining_cols):
        best_c, best_count = None, float('inf')
        m = remaining_cols
        while m:
            c = (m & -m).bit_length()-1
            cnt = 0
            for r in self.col_to_rows.get(c, []):
                if (self.rows[r] & (~remaining_cols)) == 0:
                    cnt += 1
            if cnt < best_count:
                best_count, best_c = cnt, c
                if best_count <= 1: break
            m &= m-1
        return best_c, best_count
    def solve(self, ub=float('inf')):
        self.best_solution, self.best_len = None, float('inf')
        self.search_calls = 0
        all_cols = (1 << self.N) - 1
        self._search(all_cols, [], ub)
        return self.best_solution
    def _search(self, remaining_cols, partial, ub):
        self.search_calls += 1
        # Prevent infinite recursion by limiting search calls
        if self.search_calls > self.max_search_calls:
            return
        if len(partial) >= min(self.best_len, ub): return
        if remaining_cols == 0:
            self.best_len = len(partial)
            self.best_solution = partial.copy()
            return
        c, count = self._choose_column(remaining_cols)
        if c is None or count == 0: return
        for r in self.col_to_rows.get(c, []):
            rmask = self.rows[r]
            if (rmask & (~remaining_cols)) != 0: continue
            self._search(remaining_cols & (~rmask), partial + [r], ub)

# -------------------- Undirected polyhedra --------------------

# PolyhedronGenerators in polyhedra.py exposes the undirected graph builders.

# -------------------- Solution comparison helper --------------------

def is_solution_better(candidate_solution, current_best):
    """
    Compare two solutions to determine which is better based on optimization criteria.
    
    Args:
        candidate_solution: tuple (num_endpoints, num_paths, endpoints_set, counts, dir_edges, chosen_paths)
        current_best: tuple (num_endpoints, num_paths, endpoints_set, counts, dir_edges, chosen_paths)
    
    Returns:
        True if candidate_solution is better than current_best, False otherwise
    
    Comparison criteria (in priority order):
        1. Primary: Fewer distinct endpoints (minimize num_endpoints)
        2. Secondary: If endpoints equal, fewer paths (minimize num_paths)  
        3. Tertiary: If both equal, smaller maximum per-vertex endpoint usage
    """
    if candidate_solution is None:
        return False
    if current_best is None:
        return True
        
    candidate_endpoints, candidate_paths, _, candidate_counts, _, _ = candidate_solution
    best_endpoints, best_paths, _, best_counts, _, _ = current_best
    
    # Primary criterion: fewer endpoints wins
    if candidate_endpoints < best_endpoints:
        return True
    elif candidate_endpoints > best_endpoints:
        return False
    
    # Secondary criterion: if endpoints equal, fewer paths wins
    if candidate_paths < best_paths:
        return True
    elif candidate_paths > best_paths:
        return False
    
    # Tertiary criterion: if both equal, smaller max per-vertex usage wins
    candidate_max_usage = max(candidate_counts.values()) if candidate_counts else 0
    best_max_usage = max(best_counts.values()) if best_counts else 0
    
    return candidate_max_usage < best_max_usage

# -------------------- Generic constraint-based solver --------------------

def exhaustive_orientations_with_constraints(V, undirected_edges, L, constraints=None):
    """
    Generic exhaustive orientation solver with configurable constraints.
    
    Args:
        V: Vertices
        undirected_edges: Undirected edges
        L: Path length
        constraints: List of constraint functions. Each should take (chosen_paths, dir_edges, V) 
                    and return True if constraint is satisfied.
    
    Returns:
        Best solution tuple or None
    """
    if constraints is None:
        constraints = []
    
    m = len(undirected_edges)
    best = None
    
    for mask in range(1<<m):
        dir_edges = [(u,v) if ((mask>>i)&1) else (v,u) for i,(u,v) in enumerate(undirected_edges)]
        res = solve_fixed_orientation_min_endpoints_with_L(V, undirected_edges, dir_edges, L)
        if res is None:
            continue
        
        # Apply all constraints
        chosen_paths = res[5]
        constraint_satisfied = True
        
        for constraint_func in constraints:
            if not constraint_func(chosen_paths, dir_edges, V):
                constraint_satisfied = False
                break
        
        if not constraint_satisfied:
            continue
            
        if best is None:
            best = res
        else:
            if is_solution_better(res, best):
                best = res
    
    return best

def sampled_orientations_with_constraints(V, undirected_edges, L, constraints=None, iters=1000, seed=0):
    """
    Generic sampled orientation solver with configurable constraints.
    
    Args:
        V: Vertices
        undirected_edges: Undirected edges  
        L: Path length
        constraints: List of constraint functions. Each should take (chosen_paths, dir_edges, V)
                    and return True if constraint is satisfied.
        iters: Number of iterations for sampling
        seed: Random seed
    
    Returns:
        Best solution tuple or None
    """
    if constraints is None:
        constraints = []
    
    random.seed(seed)
    m = len(undirected_edges)
    best = None
    
    for _ in range(iters):
        mask = random.getrandbits(m)
        dir_edges = [(u,v) if ((mask>>i)&1) else (v,u) for i,(u,v) in enumerate(undirected_edges)]
        res = solve_fixed_orientation_min_endpoints_with_L(V, undirected_edges, dir_edges, L)
        if res is None:
            continue
        
        # Apply all constraints
        chosen_paths = res[5]
        constraint_satisfied = True
        
        for constraint_func in constraints:
            if not constraint_func(chosen_paths, dir_edges, V):
                constraint_satisfied = False
                break
        
        if not constraint_satisfied:
            continue
            
        if best is None:
            best = res
        else:
            if is_solution_better(res, best):
                best = res
    
    return best

# -------------------- Constraint functions --------------------

def constraint_dc_only(chosen_paths, dir_edges, V):
    """Constraint: solution must be DC-only (no alternating vertices)"""
    return is_solution_dc_only(chosen_paths)

def constraint_equal_current(chosen_paths, dir_edges, V):
    """Constraint: solution must have equal current through all edges"""
    return has_equal_current(chosen_paths, dir_edges)

def constraint_sneak_free(chosen_paths, dir_edges, V):
    """Constraint: solution must be sneak-free (no shorter DC paths)"""
    return not has_sneak_paths(chosen_paths, dir_edges)

def constraint_alternating_only(chosen_paths, dir_edges, V):
    """Constraint: solution must have all alternating vertices"""
    return is_solution_alternating_only(chosen_paths)

def constraint_bipolar_only(chosen_paths, dir_edges, V):
    """Constraint: solution must be implementable with bipolar driving (no tristate Z)"""
    return is_solution_bipolar_only(chosen_paths, dir_edges, V)

# -------------------- Utility functions for constraints --------------------

def is_solution_dc_only(chosen_paths):
    """
    Check if a solution has only DC vertices (pure Anode or pure Cathode).
    Returns True if fully DC, False if any alternating vertices exist.
    """
    from collections import defaultdict
    vertex_roles = defaultdict(set)
    
    # Analyze each path to determine vertex roles
    for path in chosen_paths:
        if len(path) >= 2:
            start_vertex = path[0]
            end_vertex = path[-1]
            vertex_roles[start_vertex].add('start')
            vertex_roles[end_vertex].add('end')
    
    # Check if any vertex has both roles (alternating)
    for vertex, roles in vertex_roles.items():
        if roles == {'start', 'end'}:
            return False
    
    return True

def has_sneak_paths(chosen_paths, dir_edges):
    """
    Check if a solution has sneak paths (DC bias creates shorter paths than solver paths).
    Only meaningful for DC-only solutions.
    Returns True if sneak paths exist, False otherwise.
    """
    # First check if it's DC-only
    if not is_solution_dc_only(chosen_paths):
        return False  # Can't analyze sneak paths for non-DC solutions
    
    # Get vertex classifications
    from collections import defaultdict
    vertex_roles = defaultdict(set)
    for path in chosen_paths:
        if len(path) >= 2:
            vertex_roles[path[0]].add('start')
            vertex_roles[path[-1]].add('end')
    
    vertex_classifications = {}
    for vertex in vertex_roles.keys():
        roles = vertex_roles[vertex]
        if roles == {'start'}:
            vertex_classifications[vertex] = "Anode"
        elif roles == {'end'}:
            vertex_classifications[vertex] = "Cathode"
        else:
            vertex_classifications[vertex] = "Unknown"
    
    # Use existing sneak path analysis
    sneak_analysis = analyze_sneak_paths(chosen_paths, dir_edges, vertex_classifications)
    return sneak_analysis['has_sneak_paths']

def has_equal_current(chosen_paths, dir_edges):
    """
    Check if a solution has equal current coverage (all edges carry the same current).
    Returns True if all edges have equal current, False otherwise.
    """
    flow_data = analyze_current_flow(chosen_paths, dir_edges)
    if not flow_data:
        return True  # Vacuously true for empty solutions
    
    currents = [data['current'] for data in flow_data.values()]
    # Use string formatting to handle floating point precision
    return len(set(f"{c:.6f}" for c in currents)) == 1

def is_solution_alternating_only(chosen_paths):
    """
    Check if a solution has all alternating vertices (each vertex is both anode and cathode).
    Returns True if all vertices used in paths are alternating, False otherwise.
    """
    from collections import defaultdict
    vertex_roles = defaultdict(set)  # vertex -> set of roles ('start', 'end')
    
    # Analyze each path to determine vertex roles
    for path in chosen_paths:
        if len(path) >= 2:
            start_vertex = path[0]
            end_vertex = path[-1]
            vertex_roles[start_vertex].add('start')
            vertex_roles[end_vertex].add('end')
    
    # Check if all vertices are alternating (have both 'start' and 'end' roles)
    for vertex, roles in vertex_roles.items():
        if roles != {'start', 'end'}:
            return False  # Found a vertex that is not alternating
    
    return True

def is_solution_bipolar_only(chosen_paths, dir_edges, vertices):
    """
    Check if a solution can be implemented with a bipolar driving scheme (no tristate Z needed).

    With single-path driving (one path at a time), tristate (Z) is always required for
    inactive endpoints when there are more than 2 endpoints. Therefore, bipolar-only
    is only possible when there are exactly 2 endpoints total.

    Returns True if bipolar driving scheme is possible, False if tristate is required.
    """
    # Get all endpoint vertices
    endpoints = set()
    for path in chosen_paths:
        if len(path) >= 2:
            endpoints.add(path[0])
            endpoints.add(path[-1])

    # With single-path driving, bipolar is only possible with exactly 2 endpoints
    # (one always A, one always C, no need for Z)
    return len(endpoints) == 2

# -------------------- New objective solver --------------------

def solve_fixed_orientation_min_endpoints_with_L(V, undirected_edges, dir_edges, L):
    """
    For fixed orientation and fixed length L, among all equal-length-L exact covers:
      minimize number of distinct endpoints; tiebreaker minimize number of paths;
      then smaller max per-vertex endpoint usage.
    Return tuple:
      (num_endpoints, num_paths, endpoints_set, counts, dir_edges, chosen_paths)
    """
    m = len(undirected_edges)
    G = DiGraph(V, dir_edges)
    paths, row_masks, lens = all_geodesic_paths_dir(G)
    if not row_masks:
        return None
    # Restrict to length L
    idxs = [i for i,Lp in enumerate(lens) if Lp == L]
    if not idxs:
        return None
    rmasks = [row_masks[i] for i in idxs]
    # Quick feasibility: union must cover all edges
    union = 0
    for msk in rmasks: union |= msk
    if union != (1<<m)-1:
        return None
    # Solve exact cover
    solver = ExactCoverMinRows(m, rmasks)
    sol = solver.solve()
    if sol is None:
        return None
    chosen_rows = [idxs[i] for i in sol]
    # Endpoint stats
    endpoints = []
    for r in chosen_rows:
        P = paths[r]
        endpoints += [P[0], P[-1]]
    uniq = sorted(set(endpoints))
    cnt = Counter(endpoints)
    return (len(uniq), len(chosen_rows), uniq, cnt, dir_edges, [paths[i] for i in chosen_rows])

def analyze_sneak_paths(chosen_paths, dir_edges, vertex_classifications):
    """
    Analyze if DC bias (positive to anodes, zero to cathodes) creates
    shorter paths than the solver's chosen paths, which would "short circuit"
    the intended solution.
    
    Only applicable to fully DC designs where vertices are pure Anode or pure Cathode.
    
    Returns dict with:
    - 'has_sneak_paths': bool
    - 'solver_path_length': int 
    - 'shortest_dc_path_length': int
    - 'sneak_paths': list of shorter path info
    """
    # Build directed graph
    V = set()
    for u, v in dir_edges:
        V.add(u)
        V.add(v)
    G = DiGraph(list(V), dir_edges)
    
    # Get solver's path length (assuming all paths have same length)
    solver_path_length = len(chosen_paths[0]) - 1 if chosen_paths else 0
    
    # Identify anodes and cathodes
    anodes = [v for v, role in vertex_classifications.items() if role == "Anode"]
    cathodes = [v for v, role in vertex_classifications.items() if role == "Cathode"]
    
    # Find all shortest paths from anodes to cathodes under DC bias
    # (current can flow from any anode to any cathode)
    shortest_dc_length = float('inf')
    sneak_paths = []
    
    for anode in anodes:
        for cathode in cathodes:
            # Find shortest path from this anode to this cathode
            try:
                shortest_paths = enumerate_shortest_paths_dir(G, anode, cathode)
                if shortest_paths:
                    path_length = len(shortest_paths[0]) - 1
                    if path_length < shortest_dc_length:
                        shortest_dc_length = path_length
                        # If this is shorter than solver's path, it's a sneak path
                        if path_length < solver_path_length:
                            for path in shortest_paths:
                                sneak_paths.append({
                                    'path': path,
                                    'length': path_length,
                                    'from_anode': anode,
                                    'to_cathode': cathode
                                })
            except:
                continue
    
    if shortest_dc_length == float('inf'):
        shortest_dc_length = solver_path_length
    
    return {
        'has_sneak_paths': len(sneak_paths) > 0,
        'solver_path_length': solver_path_length,
        'shortest_dc_path_length': shortest_dc_length,
        'sneak_paths': sneak_paths
    }

def analyze_current_flow(chosen_paths, dir_edges):
    """
    Analyze current flow through edges considering all possible shortest paths
    between the same endpoints (branching analysis).
    
    Returns a dictionary mapping each directed edge to its flow data:
    {edge: {'current': float, 'paths': [path_info], 'branches': int}}
    
    When multiple shortest paths exist between the same endpoints,
    the current splits equally among all branches.
    """
    # Build directed graph
    V = set()
    for u, v in dir_edges:
        V.add(u)
        V.add(v)
    G = DiGraph(list(V), dir_edges)
    
    # Group chosen paths by their endpoints
    endpoint_groups = defaultdict(list)
    for path_idx, path in enumerate(chosen_paths):
        if len(path) >= 2:
            start, end = path[0], path[-1]
            endpoint_groups[(start, end)].append((path_idx, path))
    
    edge_flows = defaultdict(lambda: {'current': 0.0, 'paths': [], 'branches': 0})
    
    # For each endpoint pair, find ALL shortest paths and split current
    for (start, end), path_list in endpoint_groups.items():
        # Find all shortest paths between this start-end pair
        all_shortest_paths = enumerate_shortest_paths_dir(G, start, end)
        
        if not all_shortest_paths:
            continue
            
        # Current splits equally among all shortest paths
        current_per_path = 1.0 / len(all_shortest_paths)
        
        # Track which paths are actually used in the solution
        solution_paths = [path for _, path in path_list]
        
        # Add current for all possible shortest paths (branching)
        for path in all_shortest_paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge = (u, v)
                
                edge_flows[edge]['current'] += current_per_path
                edge_flows[edge]['branches'] = len(all_shortest_paths)
                
                # Mark if this path is in the actual solution
                is_solution_path = path in solution_paths
                if is_solution_path:
                    # Find which solution path this corresponds to
                    for path_idx, sol_path in path_list:
                        if sol_path == path:
                            edge_flows[edge]['paths'].append({
                                'path_id': path_idx + 1,
                                'vertices': path,
                                'is_chosen': True
                            })
                            break
                else:
                    # This is an alternative branch not chosen by solver
                    edge_flows[edge]['paths'].append({
                        'path_id': f"alt-{start}-{end}",
                        'vertices': path, 
                        'is_chosen': False
                    })
    
    # Remove duplicates in paths list for each edge
    for edge_data in edge_flows.values():
        seen = set()
        unique_paths = []
        for path_info in edge_data['paths']:
            path_key = tuple(path_info['vertices'])
            if path_key not in seen:
                seen.add(path_key)
                unique_paths.append(path_info)
        edge_data['paths'] = unique_paths
    
    return dict(edge_flows)

def solve_fixed_orientation_max_coverage_with_fixed_endpoints(V, undirected_edges, dir_edges, L, target_endpoints, dc_only=False, sneak_free=False, equal_current=False, alternating_only=False):
    """
    For a given orientation, find paths of length L that maximize edge coverage
    using exactly target_endpoints distinct endpoints.
    
    Returns (num_endpoints, num_paths, endpoints_set, counts, dir_edges, chosen_paths)
    or None if no solution with exactly target_endpoints exists.
    """
    G = DiGraph(V, dir_edges)
    all_paths, masks, lens = all_geodesic_paths_dir(G)
    
    # Filter to paths of exactly length L
    L_paths = []
    L_masks = []
    for i, length in enumerate(lens):
        if length == L:
            L_paths.append(all_paths[i])
            L_masks.append(masks[i])
    
    if not L_paths:
        return None
    
    # Find all possible combinations that use exactly target_endpoints
    from itertools import combinations
    import random
    
    # Get all unique endpoints from paths
    all_endpoints = set()
    for path in L_paths:
        all_endpoints.add(path[0])  # start
        all_endpoints.add(path[-1])  # end
    
    if len(all_endpoints) < target_endpoints:
        return None  # Not enough endpoints available
    
    best_solution = None
    max_coverage = 0
    
    # Try all combinations of target_endpoints from available endpoints
    # For efficiency, limit to reasonable number of combinations
    endpoint_combinations = list(combinations(all_endpoints, target_endpoints))
    if len(endpoint_combinations) > 1000:
        # Sample combinations if too many
        random.seed(42)
        endpoint_combinations = random.sample(endpoint_combinations, 1000)
    
    for endpoint_set in endpoint_combinations:
        endpoint_set = set(endpoint_set)
        
        # Find all paths that use only these endpoints
        valid_paths = []
        valid_masks = []
        for i, path in enumerate(L_paths):
            if path[0] in endpoint_set and path[-1] in endpoint_set:
                valid_paths.append(path)
                valid_masks.append(L_masks[i])
        
        if not valid_paths:
            continue
        
        # Use greedy approach to maximize coverage
        selected_paths = []
        selected_masks = []
        covered_edges = set()
        
        # Sort paths by number of new edges they would cover
        remaining_paths = list(zip(valid_paths, valid_masks))
        
        while remaining_paths:
            best_path = None
            best_mask = None
            best_new_edges = 0
            best_idx = -1
            
            for idx, (path, mask) in enumerate(remaining_paths):
                # Count new edges this path would add
                new_edges = 0
                for i in range(len(undirected_edges)):
                    if (mask & (1 << i)) and i not in covered_edges:
                        new_edges += 1
                
                if new_edges > best_new_edges:
                    best_new_edges = new_edges
                    best_path = path
                    best_mask = mask
                    best_idx = idx
            
            if best_new_edges == 0:
                break  # No more new edges can be covered
            
            # Add the best path
            selected_paths.append(best_path)
            selected_masks.append(best_mask)
            
            # Update covered edges
            for i in range(len(undirected_edges)):
                if best_mask & (1 << i):
                    covered_edges.add(i)
            
            # Remove the selected path
            remaining_paths.pop(best_idx)
        
        # Calculate final metrics
        total_coverage = len(covered_edges)
        if total_coverage > max_coverage:
            # Calculate endpoint usage counts
            endpoint_counts = {}
            for path in selected_paths:
                start, end = path[0], path[-1]
                endpoint_counts[start] = endpoint_counts.get(start, 0) + 1
                endpoint_counts[end] = endpoint_counts.get(end, 0) + 1
            
            candidate_solution = (
                len(endpoint_set),  # num_endpoints (should equal target_endpoints)
                len(selected_paths),  # num_paths
                endpoint_set,  # endpoints_set
                endpoint_counts,  # counts
                dir_edges,  # dir_edges
                selected_paths  # chosen_paths
            )
            
            # Apply constraint validation
            constraint_satisfied = True
            
            if dc_only and not is_solution_dc_only(selected_paths):
                constraint_satisfied = False
            if alternating_only and not is_solution_alternating_only(selected_paths):
                constraint_satisfied = False
            if equal_current and not has_equal_current(selected_paths, dir_edges):
                constraint_satisfied = False
            if sneak_free:
                # Check for sneak paths
                from collections import defaultdict
                vertex_roles = defaultdict(set)
                for path in selected_paths:
                    if len(path) >= 2:
                        vertex_roles[path[0]].add('start')
                        vertex_roles[path[-1]].add('end')
                
                vertex_classifications = {}
                for vertex in vertex_roles.keys():
                    roles = vertex_roles[vertex]
                    if roles == {'start'}:
                        vertex_classifications[vertex] = "Anode"
                    elif roles == {'end'}:
                        vertex_classifications[vertex] = "Cathode"
                    elif roles == {'start', 'end'}:
                        vertex_classifications[vertex] = "Alternating"
                    else:
                        vertex_classifications[vertex] = "Unknown"
                
                sneak_analysis = analyze_sneak_paths(selected_paths, dir_edges, vertex_classifications)
                if sneak_analysis['has_sneak_paths']:
                    constraint_satisfied = False
            
            if constraint_satisfied:
                max_coverage = total_coverage
                best_solution = candidate_solution
    
    return best_solution

def exhaustive_orientations_max_coverage_with_fixed_endpoints(V, undirected_edges, L, target_endpoints, dc_only=False, sneak_free=False, equal_current=False, alternating_only=False):
    """
    Find the orientation that maximizes edge coverage using exactly target_endpoints distinct endpoints.
    """
    if target_endpoints <= 0:
        return None
    
    m = len(undirected_edges)
    best = None
    max_coverage = 0
    
    for mask in range(1<<m):
        dir_edges = [(u,v) if ((mask>>i)&1) else (v,u) for i,(u,v) in enumerate(undirected_edges)]
        res = solve_fixed_orientation_max_coverage_with_fixed_endpoints(V, undirected_edges, dir_edges, L, target_endpoints, dc_only, sneak_free, equal_current, alternating_only)
        if res is None:
            continue
        
        # Calculate coverage (number of edges covered)
        _, _, _, _, _, chosen_paths = res
        covered_edges = set()
        for path in chosen_paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                # Find edge index
                for edge_idx, (eu, ev) in enumerate(undirected_edges):
                    if (u == eu and v == ev) or (u == ev and v == eu):
                        covered_edges.add(edge_idx)
                        break
        
        coverage = len(covered_edges)
        if coverage > max_coverage:
            max_coverage = coverage
            best = res
    
    return best

def sampled_orientations_max_coverage_with_fixed_endpoints(V, undirected_edges, L, target_endpoints, iters=1000, seed=0, dc_only=False, sneak_free=False, equal_current=False, alternating_only=False):
    """
    Sampled version: find orientation that maximizes edge coverage using exactly target_endpoints distinct endpoints.
    """
    if target_endpoints <= 0:
        return None

    random.seed(seed)
    m = len(undirected_edges)
    best = None
    max_coverage = 0

    for _ in range(iters):
        mask = random.getrandbits(m)
        dir_edges = [(u,v) if ((mask>>i)&1) else (v,u) for i,(u,v) in enumerate(undirected_edges)]
        res = solve_fixed_orientation_max_coverage_with_fixed_endpoints(V, undirected_edges, dir_edges, L, target_endpoints, dc_only, sneak_free, equal_current, alternating_only)
        if res is None:
            continue

        # Calculate coverage (number of edges covered)
        _, _, _, _, _, chosen_paths = res
        covered_edges = set()
        for path in chosen_paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                # Find edge index
                for edge_idx, (eu, ev) in enumerate(undirected_edges):
                    if (u == eu and v == ev) or (u == ev and v == eu):
                        covered_edges.add(edge_idx)
                        break

        coverage = len(covered_edges)
        if coverage > max_coverage:
            max_coverage = coverage
            best = res

    return best

# -------------------- Two-Feedpoint Solver (Edge-Geodetic Pairs) --------------------

def bfs_distances_undirected(V, undirected_edges, source):
    """
    BFS to compute shortest path distances from source in undirected graph.

    Args:
        V: List of vertices
        undirected_edges: List of undirected edges (u, v)
        source: Source vertex

    Returns:
        Dictionary mapping each vertex to its distance from source
    """
    # Build adjacency list for undirected graph
    adj = defaultdict(list)
    for u, v in undirected_edges:
        adj[u].append(v)
        adj[v].append(u)

    dist = {source: 0}
    queue = deque([source])

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)

    return dist

def all_pairs_distances(V, undirected_edges):
    """
    Compute shortest path distances between all pairs of vertices.

    Args:
        V: List of vertices
        undirected_edges: List of undirected edges

    Returns:
        Dictionary of dictionaries: dist[u][v] = shortest path length from u to v
    """
    dist = {}
    for v in V:
        dist[v] = bfs_distances_undirected(V, undirected_edges, v)
    return dist

def edge_geodetic_cover_edges(V, undirected_edges, s, t, dist):
    """
    Find edges that lie on at least one shortest s-t path.

    An edge (u,v) lies on a shortest s-t path if:
    - Both u and v are on some shortest path (dist[s][u] + dist[u][t] == D and same for v)
    - The edge connects consecutive vertices on that path (|dist[s][u] - dist[s][v]| == 1)

    Args:
        V: List of vertices
        undirected_edges: List of undirected edges
        s: Source vertex
        t: Target vertex
        dist: All-pairs distance dictionary from all_pairs_distances()

    Returns:
        Tuple (covered_edges, path_length) where:
        - covered_edges: Set of edges (as canonical tuples) covered by shortest s-t paths
        - path_length: Length of shortest s-t path (diameter between s and t)
    """
    ds = dist[s]
    dt = dist[t]
    D = ds[t]  # Distance from s to t

    covered = set()
    for u, v in undirected_edges:
        # Check if both endpoints are on some shortest s-t path
        if ds[u] + dt[u] == D and ds[v] + dt[v] == D:
            # Check if this edge is traversed on that path (consecutive vertices)
            if abs(ds[u] - ds[v]) == 1:
                # Store as canonical tuple (smaller vertex first)
                covered.add((u, v) if u < v else (v, u))

    return covered, D

def find_edge_geodetic_pairs(V, undirected_edges, require_diameter=False):
    """
    Find all vertex pairs (s, t) where all shortest paths between s and t
    together cover ALL edges in the graph.

    This is the special case for "two-feedpoint" solutions where only two
    vertices are needed as endpoints, and all edges can be lit by the
    branching shortest paths between them.

    Args:
        V: List of vertices
        undirected_edges: List of undirected edges
        require_diameter: If True, only return pairs at maximum distance (graph diameter)

    Returns:
        List of tuples (s, t, path_length) for each edge-geodetic pair found.
        Empty list if no such pairs exist (which is common for most polyhedra).
    """
    from itertools import combinations

    # Compute all-pairs shortest paths
    dist = all_pairs_distances(V, undirected_edges)

    # Get canonical edge set for comparison
    all_edges = set((u, v) if u < v else (v, u) for u, v in undirected_edges)

    # Compute graph diameter if needed
    diam = 0
    if require_diameter:
        for u in V:
            for v in V:
                if v in dist[u]:
                    diam = max(diam, dist[u][v])

    pairs = []
    for s, t in combinations(V, 2):
        covered, D = edge_geodetic_cover_edges(V, undirected_edges, s, t, dist)

        # Check if all edges are covered
        if covered == all_edges:
            # Optionally filter to diameter-only pairs
            if not require_diameter or D == diam:
                pairs.append((s, t, D))

    return pairs

def solve_two_feedpoint(V, undirected_edges, require_diameter=False):
    """
    Solve for the special two-feedpoint case where exactly two vertices
    can serve as the only feedpoints, with branching shortest paths
    covering all edges.

    This is a much simpler driving scheme than the general solver, as
    it only requires switching polarity between two fixed endpoints.

    Args:
        V: List of vertices
        undirected_edges: List of undirected edges
        require_diameter: If True, only consider pairs at maximum graph distance

    Returns:
        Dictionary with:
        - 'found': bool - whether any two-feedpoint solution exists
        - 'pairs': list of (s, t, path_length) tuples
        - 'num_edges': total number of edges in graph
        - 'diameter': graph diameter
        - 'message': human-readable summary
    """
    # Find all edge-geodetic pairs
    pairs = find_edge_geodetic_pairs(V, undirected_edges, require_diameter)

    # Compute graph diameter
    dist = all_pairs_distances(V, undirected_edges)
    diam = 0
    for u in V:
        for v in V:
            if v in dist[u]:
                diam = max(diam, dist[u][v])

    num_edges = len(undirected_edges)

    if pairs:
        if len(pairs) == 1:
            s, t, D = pairs[0]
            message = f"Found 1 two-feedpoint solution: vertices {s} and {t} (path length {D})"
        else:
            message = f"Found {len(pairs)} two-feedpoint solutions"
    else:
        message = "No two-feedpoint solution exists for this polyhedron"

    return {
        'found': len(pairs) > 0,
        'pairs': pairs,
        'num_edges': num_edges,
        'diameter': diam,
        'message': message
    }

def enumerate_two_feedpoint_paths(V, undirected_edges, s, t):
    """
    Enumerate all shortest paths between two feedpoint vertices s and t.

    Args:
        V: List of vertices
        undirected_edges: List of undirected edges
        s: First feedpoint vertex
        t: Second feedpoint vertex

    Returns:
        List of paths, where each path is a list of vertices from s to t
    """
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in undirected_edges:
        adj[u].append(v)
        adj[v].append(u)

    # BFS to get distances from s
    dist_from_s = bfs_distances_undirected(V, undirected_edges, s)

    if t not in dist_from_s:
        return []

    D = dist_from_s[t]

    # Build DAG of edges on shortest paths
    # An edge (u,v) is on a shortest path if dist[s][u] + 1 == dist[s][v] and dist[s][v] <= D
    dag = defaultdict(list)
    for u, v in undirected_edges:
        du = dist_from_s.get(u, float('inf'))
        dv = dist_from_s.get(v, float('inf'))
        if du + 1 == dv and dv <= D:
            dag[u].append(v)
        elif dv + 1 == du and du <= D:
            dag[v].append(u)

    # DFS to enumerate all paths
    paths = []
    stack = [(s, [s])]

    while stack:
        u, path = stack.pop()
        if u == t:
            paths.append(path)
            continue
        for v in dag[u]:
            stack.append((v, path + [v]))

    return paths
