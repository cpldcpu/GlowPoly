"""Interactive simulator front-end for poly_solver solvers."""

from __future__ import annotations

import math
import random
import threading
import time
import tkinter as tk
from tkinter import ttk

from polyhedra import PolyhedronGenerators
from poly_solver import (
    analyze_current_flow,
    analyze_sneak_paths,
    exhaustive_orientations_max_coverage_with_fixed_endpoints,
    sampled_orientations_max_coverage_with_fixed_endpoints,
    solve_fixed_orientation_min_endpoints_with_L,
    is_solution_better,
)
import random


def sampled_orientations_with_constraints_cancellable(V, undirected_edges, L, constraints, iters, cancel_event, seed=0):
    """
    Cancellable version of sampled orientations solver that returns partial results.
    
    Returns:
        tuple: (best_solution, iterations_completed)
        - best_solution: Best solution found so far (or None if none found)
        - iterations_completed: Number of iterations actually completed
    """
    if constraints is None:
        constraints = []
    
    random.seed(seed)
    m = len(undirected_edges)
    best = None
    iterations_completed = 0
    
    for i in range(iters):
        # Check for cancellation every 10 iterations for responsiveness
        if i % 10 == 0 and cancel_event.is_set():
            break
            
        mask = random.getrandbits(m)
        dir_edges = [(u,v) if ((mask>>i)&1) else (v,u) for i,(u,v) in enumerate(undirected_edges)]
        res = solve_fixed_orientation_min_endpoints_with_L(V, undirected_edges, dir_edges, L)
        if res is None:
            iterations_completed += 1
            continue
        
        # Apply all constraints
        chosen_paths = res[5]
        constraint_satisfied = True
        
        for constraint_func in constraints:
            if not constraint_func(chosen_paths, dir_edges, V):
                constraint_satisfied = False
                break
        
        if not constraint_satisfied:
            iterations_completed += 1
            continue
            
        if best is None:
            best = res
        else:
            if is_solution_better(res, best):
                best = res
        
        iterations_completed += 1
    
    return best, iterations_completed


def analyze_driving_schemes(chosen_paths, dir_edges, endpoints, vertices):
    """
    Analyze driving schemes for each path to determine optimal anode/cathode configurations
    that prevent unwanted LED activation.
    
    Args:
        chosen_paths: List of paths (each path is a list of vertices)
        dir_edges: List of directed edges in the solution
        endpoints: Set of endpoint vertices
        vertices: List of all vertices in the graph
    
    Returns:
        List of dictionaries, one per path, containing driving scheme analysis
    """
    analysis_results = []
    
    # Create a mapping of edges to their direction
    edge_directions = {}
    for u, v in dir_edges:
        edge_directions[(u, v)] = (u, v)  # Forward direction
        edge_directions[(v, u)] = (u, v)  # Store original direction for reverse lookup
    
    for path_idx, path in enumerate(chosen_paths):
        if len(path) < 2:
            continue
            
        start_vertex = path[0]
        end_vertex = path[-1]
        
        # Create path edges
        path_edges = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # Find the correct direction for this edge
            if (u, v) in edge_directions:
                path_edges.append((u, v))
            elif (v, u) in edge_directions:
                path_edges.append((v, u))
            else:
                path_edges.append((u, v))  # Fallback
        
        # Try to find a feasible driving scheme
        scheme_result = find_driving_scheme(
            path, path_edges, start_vertex, end_vertex, 
            endpoints, vertices, dir_edges
        )
        
        analysis_results.append({
            'path': path,
            'scheme': scheme_result
        })
    
    return analysis_results


def find_driving_scheme(target_path, target_edges, start_vertex, end_vertex, endpoints, vertices, all_dir_edges):
    """
    Find a driving scheme for a specific path that minimizes unwanted LED activation.
    
    Returns a dictionary with scheme feasibility and assignments.
    """
    # Start with the assumption: start_vertex = Anode, end_vertex = Cathode
    other_endpoints = [ep for ep in endpoints if ep not in (start_vertex, end_vertex)]
    
    # Try all combinations of A/C/Z assignments for other endpoints
    from itertools import product
    
    best_scheme = None
    min_conflicts = float('inf')
    
    # Try all possible assignments for other endpoints (A, C, or Z)
    for assignments in product(['A', 'C', 'Z'], repeat=len(other_endpoints)):
        # Create full assignment map
        full_assignment = {start_vertex: 'A', end_vertex: 'C'}
        for ep, assignment in zip(other_endpoints, assignments):
            full_assignment[ep] = assignment
        
        # Check this assignment scheme
        conflicts, blocked_paths = evaluate_scheme(
            full_assignment, target_edges, all_dir_edges, vertices
        )
        
        if len(conflicts) < min_conflicts:
            min_conflicts = len(conflicts)
            best_scheme = {
                'feasible': len(conflicts) == 0,
                'assignments': full_assignment.copy(),
                'conflicts': conflicts,
                'blocked_paths': blocked_paths,
                'required_disconnections': [ep for ep, assignment in full_assignment.items() 
                                          if assignment == 'Z' and ep in endpoints]
            }
            
            # If we found a perfect scheme, stop searching
            if len(conflicts) == 0:
                break
    
    return best_scheme or {
        'feasible': False,
        'assignments': {start_vertex: 'A', end_vertex: 'C'},
        'conflicts': [],
        'blocked_paths': [],
        'required_disconnections': other_endpoints
    }


def evaluate_scheme(assignments, target_edges, all_dir_edges, vertices):
    """
    Evaluate a driving scheme to find conflicts (unwanted LED activations).
    
    Returns:
        conflicts: List of edges that would be incorrectly activated
        blocked_paths: List of paths that are correctly blocked
    """
    conflicts = []
    blocked_paths = []
    
    # Check all directed edges to see which ones would be activated
    for u, v in all_dir_edges:
        u_assignment = assignments.get(u, 'Z')
        v_assignment = assignments.get(v, 'Z')
        
        # LED activates if u=Anode (A) and v=Cathode (C)
        edge_activated = (u_assignment == 'A' and v_assignment == 'C')
        edge_is_target = (u, v) in target_edges or (v, u) in target_edges
        
        if edge_activated and not edge_is_target:
            # This edge shouldn't be activated but would be
            conflicts.append((u, v))
        elif not edge_activated and edge_is_target:
            # This is a target edge that won't be activated (might be okay if Z is involved)
            pass
        elif not edge_activated:
            # This edge is successfully blocked
            blocked_paths.append([u, v])
    
    return conflicts, blocked_paths


def find_sneak_paths_avoided(target_path, z_endpoints, all_dir_edges):
    """
    Find sneak paths that would be created if Z endpoints were connected as A or C.
    
    Args:
        target_path: The desired path being analyzed
        z_endpoints: List of endpoints assigned to Z (unconnected)
        all_dir_edges: All directed edges in the solution
    
    Returns:
        List of sneak path edges that are avoided by the Z assignments
    """
    if not z_endpoints:
        return []
    
    sneak_paths = []
    target_start = target_path[0] if target_path else None
    target_end = target_path[-1] if target_path else None
    
    # Look for edges that would create unwanted paths if Z endpoints were connected
    for u, v in all_dir_edges:
        # Check if this edge would create a sneak path
        edge_involves_z = u in z_endpoints or v in z_endpoints
        
        if edge_involves_z:
            # This edge is blocked by having one end as Z
            # Check if connecting it would create a shorter or unwanted path
            if u in z_endpoints and v == target_end:
                # u->target_end would be a sneak path if u were made anode
                sneak_paths.append((u, v))
            elif v in z_endpoints and u == target_start:
                # target_start->v would be a sneak path if v were made cathode
                sneak_paths.append((u, v))
            elif u in z_endpoints or v in z_endpoints:
                # Any other edge involving Z endpoints could be a potential sneak
                sneak_paths.append((u, v))
    
    # Remove duplicates and limit to most relevant ones
    unique_sneaks = list(set(sneak_paths))
    return unique_sneaks[:3]  # Return up to 3 most relevant sneak paths

CANVAS_SIZE = 480
MARGIN = 50
BACKGROUND = "#f4f4f6"
ARROW_FOREGROUND_COLOR = "#ff4d4d"
ARROW_BACKGROUND_COLOR = "#b0b3c0"
EDGE_COLOR = "#1f1f24"
HIDDEN_EDGE_COLOR = "#8b8d99"
VERTEX_OUTLINE = "#2f2f35"
VERTEX_FILL_NEAR = "#ffffff"
VERTEX_FILL_FAR = "#d0d3de"
LABEL_COLOR = "#d92626"
ENDPOINT_FILL = "#ffd166"
ARROW_COLOR = "#ff4d4d"

ISO_Y = math.radians(45)
ISO_X = math.radians(35.264)
VIEW_MODES = {
    "Isometric": (ISO_Y, ISO_X),
    "Axis-Aligned": (0.0, 0.0),
    "Top-Tilt": (math.radians(20), math.radians(60)),
    "Diagonal": (math.radians(30), math.radians(135)),
    "Low-Angle": (math.radians(70), math.radians(25)),
}

POLYHEDRA = {
    # Sorted by number of edges (ascending)
    "Regular Tetrahedron": PolyhedronGenerators.undirected_tetrahedron,        # 6 edges
    "Triangular Prism": PolyhedronGenerators.undirected_triangular_prism,      # 9 edges
    "Cube": PolyhedronGenerators.undirected_cube,                              # 12 edges
    "Octahedron": PolyhedronGenerators.undirected_octahedron,                  # 12 edges
    "Pentagonal Prism": PolyhedronGenerators.undirected_pentagonal_prism,      # 15 edges
    "Truncated Tetrahedron": PolyhedronGenerators.undirected_truncated_tetrahedron, # 18 edges
    "Stellated Tetrahedron": PolyhedronGenerators.undirected_stellated_tetrahedron, # 18 edges
    "Hexagonal Prism": PolyhedronGenerators.undirected_hexagonal_prism,        # 18 edges
    "Cuboctahedron": PolyhedronGenerators.undirected_cuboctahedron,            # 24 edges
    "Rhombic Dodecahedron": PolyhedronGenerators.undirected_rhombic_dodecahedron, # 24 edges
    "Stellated Triangular Prism": PolyhedronGenerators.undirected_stellated_triangular_prism,  # 27 edges
    "Dodecahedron": PolyhedronGenerators.undirected_dodecahedron,              # 30 edges
    "Icosahedron": PolyhedronGenerators.undirected_icosahedron,                # 30 edges
    "Stellated Cube": PolyhedronGenerators.undirected_stellated_cube,          # 36 edges
    "Stellated Octahedron": PolyhedronGenerators.undirected_stellated_octahedron,  # 36 edges
    # "Rhombicuboctahedron": PolyhedronGenerators.undirected_rhombicuboctahedron, # 56 edges
    # "Snub Cube": PolyhedronGenerators.undirected_snub_cube,                    # 60 edges
    # "Rhombicosidodecahedron": PolyhedronGenerators.undirected_rhombicosidodecahedron, # 78 edges
}


def rotate_point(point, yaw, pitch):
    x, y, z = point
    x1 = x * math.cos(yaw) + z * math.sin(yaw)
    z1 = -x * math.sin(yaw) + z * math.cos(yaw)
    y2 = y * math.cos(pitch) - z1 * math.sin(pitch)
    z2 = y * math.sin(pitch) + z1 * math.cos(pitch)
    return x1, y2, z2


def project_points(coords, yaw, pitch):
    rotated = [rotate_point(p, yaw, pitch) for p in coords]
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    scale = min((CANVAS_SIZE - 2 * MARGIN) / span_x, (CANVAS_SIZE - 2 * MARGIN) / span_y)

    projected = []
    for x, y, z in rotated:
        px = (x - min_x) * scale + MARGIN
        py = (y - min_y) * scale + MARGIN
        projected.append((px, CANVAS_SIZE - py, z))
    return projected


def depth_to_color(z, z_min, z_max):
    if z_max - z_min < 1e-6:
        return VERTEX_FILL_NEAR
    t = (z - z_min) / (z_max - z_min)
    far_rgb = tuple(int(VERTEX_FILL_FAR[i : i + 2], 16) for i in (1, 3, 5))
    near_rgb = tuple(int(VERTEX_FILL_NEAR[i : i + 2], 16) for i in (1, 3, 5))
    blended = tuple(int(f + (n - f) * t) for f, n in zip(far_rgb, near_rgb))
    return "#%02x%02x%02x" % blended


def current_to_color(current, max_current):
    """Convert current flow to color from black (0) to red (max)."""
    if max_current <= 0:
        return "#000000"  # Black for no current
    
    # Normalize current to 0-1 range
    intensity = min(current / max_current, 1.0)
    
    # Interpolate from black (0,0,0) to red (255,0,0)
    red_component = int(255 * intensity)
    return f"#{red_component:02x}0000"


def draw_arrowhead(canvas, x_start, y_start, x_end, y_end, color=ARROW_FOREGROUND_COLOR, size=14):
    angle = math.atan2(y_end - y_start, x_end - x_start)
    mx = (x_start + x_end) / 2
    my = (y_start + y_end) / 2
    spread = math.radians(25)
    points = [
        mx, my,
        mx - size * math.cos(angle - spread), my - size * math.sin(angle - spread),
        mx - size * math.cos(angle + spread), my - size * math.sin(angle + spread),
    ]
    canvas.create_polygon(points, fill=color, outline=color)


def draw_polyhedron(canvas, builder, active_view, highlight=None, current_data=None):
    canvas.delete("all")
    canvas.create_rectangle(0, 0, CANVAS_SIZE, CANVAS_SIZE, fill=BACKGROUND, outline=BACKGROUND)

    _, edges, coords = builder(return_coords=True)
    projected = project_points(coords, *active_view)
    z_vals = [p[2] for p in projected]
    z_min, z_max = min(z_vals), max(z_vals)
    depth_threshold = sum(z_vals) / len(z_vals)

    highlight = highlight or {}
    endpoint_indices = set(highlight.get('endpoints', []))
    oriented_edges = highlight.get('edges', [])
    all_directed_edges = highlight.get('all_directed_edges', [])
    
    # Create orientation map from solution edges (for highlighting)
    solution_orientation_map = {tuple(sorted((u, v))): (u, v) for u, v in oriented_edges}
    
    # Create complete orientation map from all directed edges (for arrows)
    complete_orientation_map = {tuple(sorted((u, v))): (u, v) for u, v in all_directed_edges}

    def edge_depth(edge):
        u, v = edge
        return (projected[u][2] + projected[v][2]) / 2

    # Calculate maximum current for color scaling
    max_current = 0
    if current_data:
        max_current = max(data['current'] for data in current_data.values()) if current_data else 0

    sorted_edges = sorted(edges, key=edge_depth)
    for u, v in sorted_edges:
        x1, y1, z1 = projected[u]
        x2, y2, z2 = projected[v]
        depth = (z1 + z2) / 2
        
        # Determine edge color based on current flow
        edge_key = (u, v)
        if current_data and edge_key in current_data:
            current = current_data[edge_key]['current']
            edge_color = current_to_color(current, max_current)
        elif current_data and (v, u) in current_data:
            current = current_data[(v, u)]['current']
            edge_color = current_to_color(current, max_current)
        else:
            # No current data or edge not in solution
            edge_color = HIDDEN_EDGE_COLOR if depth > depth_threshold else EDGE_COLOR
        
        if depth > depth_threshold:
            canvas.create_line(x1, y1, x2, y2, fill=edge_color, dash=(6, 4), width=2)

    for u, v in sorted_edges:
        x1, y1, z1 = projected[u]
        x2, y2, z2 = projected[v]
        depth = (z1 + z2) / 2
        
        # Determine edge color based on current flow
        edge_key = (u, v)
        if current_data and edge_key in current_data:
            current = current_data[edge_key]['current']
            edge_color = current_to_color(current, max_current)
        elif current_data and (v, u) in current_data:
            current = current_data[(v, u)]['current']
            edge_color = current_to_color(current, max_current)
        else:
            # No current data or edge not in solution
            edge_color = HIDDEN_EDGE_COLOR if depth > depth_threshold else EDGE_COLOR
        
        if depth <= depth_threshold:
            canvas.create_line(x1, y1, x2, y2, fill=edge_color, width=3, capstyle=tk.ROUND)

    # Draw arrows for all edges that have an orientation
    for u, v in edges:
        edge_key = (u, v)
        has_current = (current_data and edge_key in current_data) or (current_data and (v, u) in current_data)
        key = tuple(sorted((u, v)))
        
        # Use complete orientation map to show arrows on all edges
        orient = complete_orientation_map.get(key)
        
        if orient:
            x_start, y_start = projected[orient[0]][0], projected[orient[0]][1]
            x_end, y_end = projected[orient[1]][0], projected[orient[1]][1]
            
            # Use different arrow properties based on current flow and depth
            depth = (projected[u][2] + projected[v][2]) / 2
            arrow_size = 16 if depth <= depth_threshold else 14
            
            # Different arrow colors: red for current-carrying, gray for non-current
            if has_current:
                arrow_color = ARROW_FOREGROUND_COLOR  # Red for current-carrying edges
            else:
                arrow_color = "#888888"  # Gray for non-current edges
                
            draw_arrowhead(canvas, x_start, y_start, x_end, y_end, color=arrow_color, size=arrow_size)

    radius = 6
    for idx, (x, y, z) in enumerate(projected):
        fill = ENDPOINT_FILL if idx in endpoint_indices else depth_to_color(z, z_min, z_max)
        canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            outline=VERTEX_OUTLINE,
            width=2,
            fill=fill,
        )
        canvas.create_text(x, y - radius - 8, text=str(idx), fill=LABEL_COLOR, font=("Helvetica", 11, "bold"))


def build_constraint_suffix(dc_only: bool, sneak_free: bool, equal_current: bool, alternating_only: bool, bipolar_only: bool) -> str:
    """Build constraint suffix string for result formatting."""
    constraints = []
    if dc_only:
        constraints.append("DC only")
    if sneak_free:
        constraints.append("sneak free")
    if equal_current:
        constraints.append("equal current")
    if alternating_only:
        constraints.append("alternating only")
    if bipolar_only:
        constraints.append("bipolar only")
    return f" ({', '.join(constraints)})" if constraints else ""


def format_result_header(name: str, L, res, constraint_suffix: str, fixed_endpoints: int = 0, path_mode: str = "fixed") -> list:
    """Format the header section of the result."""
    if res is None:
        if fixed_endpoints > 0:
            return [f"=== {name} | L={L}{constraint_suffix} ===", f"No solution found with exactly {fixed_endpoints} endpoints.", ""]
        else:
            return [f"=== {name} | L={L}{constraint_suffix} ===", "No equal-length-L exact cover found.", ""]
    
    n_end, n_paths, endpoints, counts, dir_edges, chosen_paths = res
    
    if fixed_endpoints > 0:
        # Calculate edge coverage for fixed endpoints mode
        covered_edges = set()
        for path in chosen_paths:
            for i in range(len(path) - 1):
                covered_edges.add((path[i], path[i + 1]))
        
        return [
            f"=== {name} | L={L}: maximize coverage with {fixed_endpoints} endpoints{constraint_suffix} ===",
            f"Fixed endpoints: {n_end} | Paths: {n_paths} | Edge coverage: {len(covered_edges)}",
            f"Endpoint set: {endpoints}",
            f"Endpoint usage counts: {dict(sorted(counts.items()))}",
        ]
    else:
        # Calculate edge coverage for all non-fixed modes
        covered_edges = set()
        for path in chosen_paths:
            for i in range(len(path) - 1):
                covered_edges.add((path[i], path[i + 1]))
        
        if path_mode == "variable":
            return [
                f"=== {name} | Variable Path: maximize coverage, minimize endpoints{constraint_suffix} ===",
                f"Best path length: {L} | Endpoints: {n_end} | Paths: {n_paths} | Edge coverage: {len(covered_edges)}",
                f"Endpoint set: {endpoints}",
                f"Endpoint usage counts: {dict(sorted(counts.items()))}",
            ]
        elif path_mode == "mixed":
            return [
                f"=== {name} | Mixed Path: maximize coverage, minimize endpoints{constraint_suffix} ===",
                f"Path lengths: {L} | Endpoints: {n_end} | Paths: {n_paths} | Edge coverage: {len(covered_edges)}",
                f"Endpoint set: {endpoints}",
                f"Endpoint usage counts: {dict(sorted(counts.items()))}",
            ]
        else:  # fixed mode
            return [
                f"=== {name} | L={L}: minimize distinct endpoints{constraint_suffix} ===",
                f"Distinct endpoints: {n_end} | Paths: {n_paths}",
                f"Endpoint set: {endpoints}",
                f"Endpoint usage counts: {dict(sorted(counts.items()))}",
            ]


def format_path_details(chosen_paths, dir_edges, total_edges: int) -> list:
    """Format path listing and coverage details."""
    lines = []
    used = 0
    for idx, path in enumerate(chosen_paths, 1):
        edges = list(zip(path, path[1:]))
        used += len(path) - 1
        lines.append(f"  Path {idx}: vertices {path} | len={len(path) - 1} | edges={edges}")
    lines.append(f"Total edges covered: {used}")
    
    # Add coverage ratio if below complete coverage
    if total_edges > 0 and used < total_edges:
        coverage_ratio = used / total_edges
        lines.append(f"Coverage ratio: {coverage_ratio:.3f} ({used}/{total_edges})")
    
    lines.append(f"Orientation (directed edges): {sorted(dir_edges)}")
    return lines


def format_branching_analysis(chosen_paths, dir_edges) -> list:
    """Format branching analysis section."""
    lines = ["", "=== Branching Analysis ==="]
    flow_data = analyze_current_flow(chosen_paths, dir_edges)
    
    # Test for equal current coverage
    currents = [data['current'] for data in flow_data.values()]
    equal_current = len(set(f"{c:.6f}" for c in currents)) == 1 if currents else False
    uniform_current = currents[0] if equal_current and currents else None
    
    if equal_current:
        lines.append(f"Equal current coverage: YES (all edges carry {uniform_current:.3f} current)")
    else:
        min_current = min(currents) if currents else 0
        max_current = max(currents) if currents else 0
        lines.append(f"Equal current coverage: NO (range {min_current:.3f} to {max_current:.3f})")
    lines.append("")
    
    # Sort edges by their total current (descending) for better readability
    sorted_edges = sorted(flow_data.items(), key=lambda x: x[1]['current'], reverse=True)
    
    for edge, data in sorted_edges:
        current = data['current']
        branches = data['branches']
        paths = data['paths']
        
        # Extract start-end pairs from all paths traversing this edge
        path_endpoints = []
        for path_info in paths:
            vertices = path_info['vertices']
            if len(vertices) >= 2:
                start, end = vertices[0], vertices[-1]
                endpoint_pair = f"[{start},{end}]"
                if endpoint_pair not in path_endpoints:
                    path_endpoints.append(endpoint_pair)
        
        paths_str = " ".join(path_endpoints)
        lines.append(f"Edge {edge}: current={current:.3f}, branches={branches}, paths={paths_str}")
    
    return lines


def format_polarity_analysis(chosen_paths) -> tuple[list, dict]:
    """Format polarity analysis section and return vertex classifications."""
    lines = ["", "=== Polarity Analysis ==="]
    from collections import defaultdict
    vertex_roles = defaultdict(set)  # vertex -> set of roles ('start', 'end')
    
    # Analyze each path to determine vertex roles
    for path in chosen_paths:
        if len(path) >= 2:
            start_vertex = path[0]
            end_vertex = path[-1]
            vertex_roles[start_vertex].add('start')
            vertex_roles[end_vertex].add('end')
    
    # Classify each vertex and check for fully DC design
    fully_dc = True
    vertex_classifications = {}
    
    for vertex in sorted(vertex_roles.keys()):
        roles = vertex_roles[vertex]
        if roles == {'start'}:
            classification = "Anode"
        elif roles == {'end'}:
            classification = "Cathode"
        elif roles == {'start', 'end'}:
            classification = "Alternating"
            fully_dc = False
        else:
            classification = "Unknown"
            fully_dc = False
        
        vertex_classifications[vertex] = classification
        lines.append(f"Vertex {vertex}: {classification}")
    
    if fully_dc:
        lines.append("Design classification: Fully DC (all vertices are either pure Anode or pure Cathode)")
    else:
        lines.append("Design classification: Mixed polarity (contains Alternating vertices)")
    
    return lines, vertex_classifications


def format_sneak_path_analysis(chosen_paths, dir_edges, vertex_classifications) -> list:
    """Format sneak path analysis section for fully DC designs."""
    lines = ["", "=== Sneak Path Analysis ==="]
    sneak_analysis = analyze_sneak_paths(chosen_paths, dir_edges, vertex_classifications)
    
    if sneak_analysis['has_sneak_paths']:
        lines.append("SNEAK PATHS DETECTED: DC bias creates shorter paths than solver solution!")
        lines.append(f"Solver path length: {sneak_analysis['solver_path_length']}")
        lines.append(f"Shortest DC path length: {sneak_analysis['shortest_dc_path_length']}")
        lines.append("Sneak paths found:")
        for i, path_info in enumerate(sneak_analysis['sneak_paths'], 1):
            path_str = " -> ".join(map(str, path_info['path']))
            lines.append(f"  Sneak path {i}: {path_str} (length {path_info['length']})")
    else:
        lines.append("NO SNEAK PATHS: DC bias does not create shorter paths than solver solution")
        lines.append(f"All paths maintain minimum length of {sneak_analysis['solver_path_length']}")
    
    return lines


def format_driving_scheme_analysis(chosen_paths, dir_edges, endpoints) -> list:
    """Format driving scheme analysis section."""
    lines = ["", "=== Driving Scheme Analysis ==="]
    
    # Derive vertices from paths and edges
    all_vertices = set()
    for path in chosen_paths:
        all_vertices.update(path)
    for u, v in dir_edges:
        all_vertices.add(u)
        all_vertices.add(v)
    vertices_list = sorted(list(all_vertices))
    
    driving_analysis = analyze_driving_schemes(chosen_paths, dir_edges, endpoints, vertices_list)
    
    for i, path_analysis in enumerate(driving_analysis, 1):
        path = path_analysis['path']
        scheme = path_analysis['scheme']
        
        # Create compact endpoint assignment string
        all_endpoints_sorted = sorted(endpoints)
        assignments = []
        z_endpoints = []
        for ep in all_endpoints_sorted:
            assignment = scheme['assignments'].get(ep, 'Z')
            assignments.append(f"{ep}={assignment}")
            if assignment == 'Z':
                z_endpoints.append(ep)
        
        endpoint_str = ' '.join(assignments)
        
        # Determine result status and identify sneak paths avoided by Z assignments
        if scheme['feasible']:
            if z_endpoints:
                # Find sneak paths that would be created if Z endpoints were connected
                sneak_paths = find_sneak_paths_avoided(path, z_endpoints, dir_edges)
                if sneak_paths:
                    sneak_comment = f" # avoids sneak: {' '.join(f'{u}->{v}' for u, v in sneak_paths[:2])}"
                    if len(sneak_paths) > 2:
                        sneak_comment += f" +{len(sneak_paths)-2} more"
                else:
                    sneak_comment = ""
                result_str = sneak_comment
            else:
                result_str = ""  # Nothing if proper configuration found
        elif any(assignment.endswith('=Z') for assignment in assignments):
            result_str = " - Tristate required"
        else:
            result_str = " - Impossible"
        
        lines.append(f"Path {i}: {' -> '.join(map(str, path))} | {endpoint_str}{result_str}")
    
    # Add final conclusion about driving scheme type
    lines.append("")
    has_tristate = any('Z' in analysis['scheme']['assignments'].values() for analysis in driving_analysis)
    has_feasible = any(analysis['scheme']['feasible'] for analysis in driving_analysis)
    all_infeasible = all(not analysis['scheme']['feasible'] for analysis in driving_analysis)
    
    if all_infeasible:
        conclusion = "No driving scheme found"
    elif has_tristate:
        conclusion = "Ternary driving scheme found"
    else:
        conclusion = "Bipolar driving scheme found"
    
    lines.append(conclusion)
    return lines


def format_result(name: str, L, res, total_edges: int = 0, dc_only: bool = False, sneak_free: bool = False, equal_current: bool = False, alternating_only: bool = False, bipolar_only: bool = False, fixed_endpoints: int = 0, path_mode: str = "fixed") -> str:
    constraint_suffix = build_constraint_suffix(dc_only, sneak_free, equal_current, alternating_only, bipolar_only)
    
    # Handle no solution case
    if res is None:
        if fixed_endpoints > 0:
            return f"=== {name} | L={L}{constraint_suffix} ===\nNo solution found with exactly {fixed_endpoints} endpoints.\n"
        else:
            return f"=== {name} | L={L}{constraint_suffix} ===\nNo equal-length-L exact cover found.\n"
    
    # Get result components
    n_end, n_paths, endpoints, counts, dir_edges, chosen_paths = res
    
    # Format header section
    lines = format_result_header(name, L, res, constraint_suffix, fixed_endpoints, path_mode)
    
    # Add path listing and coverage info
    lines.extend(format_path_details(chosen_paths, dir_edges, total_edges))
    
    # Add branching analysis
    lines.extend(format_branching_analysis(chosen_paths, dir_edges))
    
    # Add polarity analysis
    polarity_lines, vertex_classifications = format_polarity_analysis(chosen_paths)
    lines.extend(polarity_lines)
    
    # Add sneak path analysis for fully DC designs
    fully_dc = all(classification in ["Anode", "Cathode"] for classification in vertex_classifications.values())
    if fully_dc:
        lines.extend(format_sneak_path_analysis(chosen_paths, dir_edges, vertex_classifications))
    
    # Add driving scheme analysis
    lines.extend(format_driving_scheme_analysis(chosen_paths, dir_edges, endpoints))
    
    return "\n".join(lines) + "\n"


def run_variable_path_solver(name, builder, iters, cancel_event, dc_only=False, sneak_free=False, equal_current=False, alternating_only=False, bipolar_only=False):
    """
    Variable path length solver: maximize coverage while minimizing endpoints.
    Tries different path lengths and finds the best combination.
    """
    V, E = builder()
    total_edges = len(E)
    start = time.time()
    
    # Check for cancellation before starting computation
    if cancel_event.is_set():
        return None
    
    # Build constraints list dynamically
    constraints = []
    if dc_only:
        from poly_solver import constraint_dc_only
        constraints.append(constraint_dc_only)
    if sneak_free:
        from poly_solver import constraint_sneak_free
        constraints.append(constraint_sneak_free)
    if equal_current:
        from poly_solver import constraint_equal_current
        constraints.append(constraint_equal_current)
    if alternating_only:
        from poly_solver import constraint_alternating_only
        constraints.append(constraint_alternating_only)
    if bipolar_only:
        from poly_solver import constraint_bipolar_only
        constraints.append(constraint_bipolar_only)
    
    # Build constraint list for mode description
    constraint_names = []
    if dc_only:
        constraint_names.append("DC only")
    if sneak_free:
        constraint_names.append("sneak free")
    if equal_current:
        constraint_names.append("equal current")
    if alternating_only:
        constraint_names.append("alternating only")
    if bipolar_only:
        constraint_names.append("bipolar only")
    constraint_suffix = f" ({', '.join(constraint_names)})" if constraint_names else ""
    
    best_solution = None
    best_coverage = 0
    best_L = 0
    iterations_completed = 0
    
    # Try different path lengths (typically 1 to 6 should cover most cases)
    max_L = min(6, len(V) - 1)  # Don't exceed reasonable path lengths
    
    for L in range(1, max_L + 1):
        # Check for cancellation
        if cancel_event.is_set():
            break
            
        # Use appropriate solver based on graph size
        if len(E) <= 12:
            from poly_solver import exhaustive_orientations_with_constraints
            res = exhaustive_orientations_with_constraints(V, E, L, constraints=constraints)
            iterations_for_L = "exhaustive"
        else:
            # For larger graphs, use fewer iterations per L to keep total time reasonable
            iters_per_L = max(200, iters // max_L)
            res, iters_completed = sampled_orientations_with_constraints_cancellable(V, E, L, constraints, iters_per_L, cancel_event, seed=42 + L)
            iterations_completed += iters_completed
            iterations_for_L = iters_completed
            
            # If cancelled during this L, break
            if cancel_event.is_set() and res is None:
                break
        
        if res is not None:
            # Calculate coverage for this solution
            _, _, _, _, _, chosen_paths = res
            covered_edges = set()
            for path in chosen_paths:
                for i in range(len(path) - 1):
                    covered_edges.add((path[i], path[i + 1]))
            
            coverage = len(covered_edges)
            endpoints = res[0]  # Number of distinct endpoints
            
            # Evaluation criteria: maximize coverage, then minimize endpoints
            is_better = False
            if best_solution is None:
                is_better = True
            elif coverage > best_coverage:
                is_better = True
            elif coverage == best_coverage and endpoints < best_solution[0]:
                is_better = True
            
            if is_better:
                best_solution = res
                best_coverage = coverage
                best_L = L
    
    duration = time.time() - start
    
    if best_solution is None:
        return None
    
    # Determine mode description
    if cancel_event.is_set():
        mode = f"variable path (L=1-{max_L}, STOPPED)" + constraint_suffix
    else:
        if len(E) <= 12:
            mode = f"variable path exhaustive (best L={best_L})" + constraint_suffix
        else:
            mode = f"variable path sampled ({iterations_completed} total iterations, best L={best_L})" + constraint_suffix
    
    return best_solution, mode, duration, total_edges


def solve_mixed_path_exact_cover(V, undirected_edges, dir_edges, constraints):
    """
    Mixed path length exact cover solver.
    Unlike the fixed-length solver, this allows paths of different lengths in the same solution.
    """
    from poly_solver import DiGraph, all_geodesic_paths_dir, ExactCoverMinRows
    from collections import Counter
    
    m = len(undirected_edges)
    G = DiGraph(V, dir_edges)
    
    # Get ALL geodesic paths (all lengths) instead of filtering to specific length
    all_paths, row_masks, path_lengths = all_geodesic_paths_dir(G)
    
    if not row_masks:
        return None
    
    # Quick feasibility: union must cover all edges
    union = 0
    for mask in row_masks:
        union |= mask
    if union != (1<<m)-1:
        return None
    
    # Solve exact cover with all path lengths allowed
    # Use more aggressive limits for mixed path as it's much more expensive
    solver = ExactCoverMinRows(m, row_masks)
    if len(row_masks) > 100:  # If too many paths, use very tight limit
        solver.max_search_calls = 50000  # Very restrictive for complex cases
    elif len(row_masks) > 50:
        solver.max_search_calls = 200000  # Moderately restrictive
    sol = solver.solve()
    if sol is None:
        return None
    
    chosen_paths = [all_paths[i] for i in sol]
    
    # Apply constraints to the mixed-length solution
    if constraints:
        constraint_satisfied = True
        for constraint_func in constraints:
            if not constraint_func(chosen_paths, dir_edges, V):
                constraint_satisfied = False
                break
        
        if not constraint_satisfied:
            return None
    
    # Calculate endpoint statistics
    endpoints = []
    for path in chosen_paths:
        endpoints += [path[0], path[-1]]
    
    unique_endpoints = sorted(set(endpoints))
    endpoint_counts = Counter(endpoints)
    
    return (len(unique_endpoints), len(chosen_paths), unique_endpoints, endpoint_counts, dir_edges, chosen_paths)


def run_mixed_path_solver(name, builder, iters, cancel_event, dc_only=False, sneak_free=False, equal_current=False, alternating_only=False, bipolar_only=False):
    """
    Mixed path length solver: allows paths of different lengths within the same solution.
    Maximizes coverage while minimizing endpoints by considering all possible path lengths simultaneously.
    """
    V, E = builder()
    total_edges = len(E)
    start = time.time()
    
    # Check for cancellation before starting computation
    if cancel_event.is_set():
        return None
    
    # Build constraints list dynamically
    constraints = []
    if dc_only:
        from poly_solver import constraint_dc_only
        constraints.append(constraint_dc_only)
    if sneak_free:
        from poly_solver import constraint_sneak_free
        constraints.append(constraint_sneak_free)
    if equal_current:
        from poly_solver import constraint_equal_current
        constraints.append(constraint_equal_current)
    if alternating_only:
        from poly_solver import constraint_alternating_only
        constraints.append(constraint_alternating_only)
    if bipolar_only:
        from poly_solver import constraint_bipolar_only
        constraints.append(constraint_bipolar_only)
    
    # Build constraint list for mode description
    constraint_names = []
    if dc_only:
        constraint_names.append("DC only")
    if sneak_free:
        constraint_names.append("sneak free")
    if equal_current:
        constraint_names.append("equal current")
    if alternating_only:
        constraint_names.append("alternating only")
    if bipolar_only:
        constraint_names.append("bipolar only")
    constraint_suffix = f" ({', '.join(constraint_names)})" if constraint_names else ""
    
    # This is a more complex optimization problem. We need to:
    # 1. Generate all possible geodesic paths of all reasonable lengths
    # 2. Use optimization to find the best combination that maximizes coverage
    
    # For now, implement a simplified version that uses sampling across different orientations
    # and allows the exact cover solver to pick from paths of different lengths
    
    m = len(E)
    best_solution = None
    best_coverage = 0
    iterations_completed = 0
    
    # Use fewer iterations since this is more expensive
    max_iters = min(iters, 1000) if len(E) > 12 else 1 << m
    
    for iteration in range(max_iters):
        # Check for cancellation
        if cancel_event.is_set():
            break
            
        # Early termination if individual iterations are taking too long
        iteration_start = time.time()
        
        if len(E) <= 12:
            # Exhaustive for small graphs
            mask = iteration
            if mask >= (1 << m):
                break
        else:
            # Random sampling for large graphs
            mask = random.getrandbits(m)
        
        dir_edges = [(u,v) if ((mask>>i)&1) else (v,u) for i,(u,v) in enumerate(E)]
        
        # Try mixed-length exact cover solver
        res = solve_mixed_path_exact_cover(V, E, dir_edges, constraints)
        
        # Check if this iteration took too long
        iteration_time = time.time() - iteration_start
        if iteration_time > 5.0:  # If single iteration takes more than 5 seconds
            print(f"Warning: Mixed path iteration {iteration} took {iteration_time:.1f}s, may need early termination")
        
        if res is not None:
            # Calculate coverage for this solution
            _, _, _, _, _, chosen_paths = res
            covered_edges = set()
            for path in chosen_paths:
                for i in range(len(path) - 1):
                    covered_edges.add((path[i], path[i + 1]))
            
            coverage = len(covered_edges)
            endpoints = res[0]  # Number of distinct endpoints
            
            # Evaluation criteria: maximize coverage, then minimize endpoints
            is_better = False
            if best_solution is None:
                is_better = True
            elif coverage > best_coverage:
                is_better = True
            elif coverage == best_coverage and endpoints < best_solution[0]:
                is_better = True
            
            if is_better:
                best_solution = res
                best_coverage = coverage
        
        iterations_completed += 1
    
    duration = time.time() - start
    
    if best_solution is None:
        return None
    
    # Determine mode description
    if cancel_event.is_set():
        mode = f"mixed path (STOPPED after {iterations_completed} iterations)" + constraint_suffix
    else:
        if len(E) <= 12:
            mode = f"mixed path exhaustive" + constraint_suffix
        else:
            mode = f"mixed path sampled ({iterations_completed} iterations)" + constraint_suffix
    
    return best_solution, mode, duration, total_edges


def run_solver(name, builder, L, iters, cancel_event, dc_only=False, sneak_free=False, equal_current=False, alternating_only=False, bipolar_only=False, fixed_endpoints=0):
    V, E = builder()
    total_edges = len(E)
    start = time.time()
    
    # Check for cancellation before starting computation
    if cancel_event.is_set():
        return None
    
    # Check if using fixed endpoints constraint (overrides minimize objective)
    if fixed_endpoints > 0:
        # Build constraint list for mode description
        constraints = []
        if dc_only:
            constraints.append("DC only")
        if sneak_free:
            constraints.append("sneak free")
        if equal_current:
            constraints.append("equal current")
        if alternating_only:
            constraints.append("alternating only")
        if bipolar_only:
            constraints.append("bipolar only")
        constraint_suffix = f" ({', '.join(constraints)})" if constraints else ""
        
        if len(E) <= 12:
            res = exhaustive_orientations_max_coverage_with_fixed_endpoints(V, E, L, fixed_endpoints, dc_only, sneak_free, equal_current, alternating_only)
            mode = f"exhaustive (fixed {fixed_endpoints} endpoints){constraint_suffix}"
            iterations_completed = "all"
        else:
            # For now, use the existing function for fixed endpoints
            # TODO: Could implement cancellable version for fixed endpoints too
            res = sampled_orientations_max_coverage_with_fixed_endpoints(V, E, L, fixed_endpoints, iters=iters, seed=42, dc_only=dc_only, sneak_free=sneak_free, equal_current=equal_current, alternating_only=alternating_only)
            if cancel_event.is_set():
                return None  # Early cancellation for fixed endpoints
            mode = f"sampled ({iters} iterations, fixed {fixed_endpoints} endpoints){constraint_suffix}"
            iterations_completed = iters
            
        duration = time.time() - start
        return res, mode, duration, total_edges
    
    # Original minimize endpoints logic
    constraints = []
    if dc_only:
        constraints.append("DC only")
    if sneak_free:
        constraints.append("sneak free")
    if equal_current:
        constraints.append("equal current")
    if alternating_only:
        constraints.append("alternating only")
    if bipolar_only:
        constraints.append("bipolar only")
    constraint_suffix = f" ({', '.join(constraints)})" if constraints else ""
    
    # Build constraints list dynamically
    constraints = []
    if dc_only:
        from poly_solver import constraint_dc_only
        constraints.append(constraint_dc_only)
    if sneak_free:
        from poly_solver import constraint_sneak_free
        constraints.append(constraint_sneak_free)
    if equal_current:
        from poly_solver import constraint_equal_current
        constraints.append(constraint_equal_current)
    if alternating_only:
        from poly_solver import constraint_alternating_only
        constraints.append(constraint_alternating_only)
    if bipolar_only:
        from poly_solver import constraint_bipolar_only
        constraints.append(constraint_bipolar_only)
    
    # Use generic constraint-based solvers
    if len(E) <= 12:
        from poly_solver import exhaustive_orientations_with_constraints
        res = exhaustive_orientations_with_constraints(V, E, L, constraints=constraints)
        mode = "exhaustive" + constraint_suffix
        iterations_completed = "all"  # Exhaustive always completes all possibilities
    else:
        # Use cancellable sampled solver
        res, iterations_completed = sampled_orientations_with_constraints_cancellable(V, E, L, constraints, iters, cancel_event, seed=42)
        if cancel_event.is_set() and res is not None:
            mode = f"sampled ({iterations_completed}/{iters} iterations, STOPPED)" + constraint_suffix
        else:
            mode = f"sampled ({iters} iterations)" + constraint_suffix
            iterations_completed = iters
    
    duration = time.time() - start
    return res, mode, duration, total_edges


def create_main_window():
    """Create and configure the main window."""
    root = tk.Tk()
    root.title("Glowing Polyhedron Simulator")
    root.configure(bg="#e8e8ed")
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    return root


def render_markdown_to_text(text_widget, markdown_content):
    """Render basic Markdown formatting in a Text widget."""
    import re
    
    # Define text tags for formatting
    text_widget.tag_configure("h1", font=("Arial", 16, "bold"), spacing1=20, spacing3=10, foreground="#2c3e50")
    text_widget.tag_configure("h2", font=("Arial", 14, "bold"), spacing1=15, spacing3=8, foreground="#34495e")
    text_widget.tag_configure("h3", font=("Arial", 12, "bold"), spacing1=10, spacing3=5, foreground="#7f8c8d")
    text_widget.tag_configure("bold", font=("Arial", 10, "bold"))
    text_widget.tag_configure("italic", font=("Arial", 10, "italic"))
    text_widget.tag_configure("code", font=("Consolas", 9), background="#f1f2f6", relief="solid", borderwidth=1)
    text_widget.tag_configure("code_block", font=("Consolas", 9), background="#f8f9fa", relief="solid", borderwidth=1, lmargin1=20, lmargin2=20)
    text_widget.tag_configure("normal", font=("Arial", 10))
    text_widget.tag_configure("bullet", lmargin1=20, lmargin2=40)
    
    lines = markdown_content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Handle code blocks
        if line.strip().startswith('```'):
            i += 1
            code_content = []
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_content.append(lines[i])
                i += 1
            if code_content:
                text_widget.insert(tk.END, '\n'.join(code_content) + '\n\n', "code_block")
            i += 1
            continue
        
        # Handle headers
        if line.startswith('# '):
            text_widget.insert(tk.END, line[2:] + '\n', "h1")
        elif line.startswith('## '):
            text_widget.insert(tk.END, line[3:] + '\n', "h2")
        elif line.startswith('### '):
            text_widget.insert(tk.END, line[4:] + '\n', "h3")
        else:
            # Handle inline formatting
            if line.strip():
                formatted_line = line
                
                # Handle bullet points
                if line.strip().startswith('- ') or line.strip().startswith('* '):
                    bullet_text = line.strip()[2:]
                    text_widget.insert(tk.END, f"• {bullet_text}\n", "bullet")
                else:
                    # Process inline formatting (bold, italic, code)
                    # Use a more careful approach to preserve spaces
                    remaining_text = formatted_line
                    
                    # Process formatting in order of priority (bold first, then italic, then code)
                    # Bold text: **text**
                    remaining_text = re.sub(r'\*\*(.*?)\*\*', lambda m: f'\x00BOLD_START\x00{m.group(1)}\x00BOLD_END\x00', remaining_text)
                    # Italic text: *text* (but not if it's part of bold markers)
                    remaining_text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', lambda m: f'\x00ITALIC_START\x00{m.group(1)}\x00ITALIC_END\x00', remaining_text)
                    # Code text: `text`
                    remaining_text = re.sub(r'`([^`]+)`', lambda m: f'\x00CODE_START\x00{m.group(1)}\x00CODE_END\x00', remaining_text)
                    
                    # Split by our markers and insert with appropriate tags
                    parts = re.split(r'\x00(BOLD_START|BOLD_END|ITALIC_START|ITALIC_END|CODE_START|CODE_END)\x00', remaining_text)
                    
                    current_tag = "normal"
                    tag_stack = []
                    
                    for part in parts:
                        if part == "BOLD_START":
                            tag_stack.append(current_tag)
                            current_tag = "bold"
                        elif part == "BOLD_END":
                            current_tag = tag_stack.pop() if tag_stack else "normal"
                        elif part == "ITALIC_START":
                            tag_stack.append(current_tag)
                            current_tag = "italic"
                        elif part == "ITALIC_END":
                            current_tag = tag_stack.pop() if tag_stack else "normal"
                        elif part == "CODE_START":
                            tag_stack.append(current_tag)
                            current_tag = "code"
                        elif part == "CODE_END":
                            current_tag = tag_stack.pop() if tag_stack else "normal"
                        elif part:  # Regular text content
                            text_widget.insert(tk.END, part, current_tag)
                    
                    text_widget.insert(tk.END, '\n', "normal")
            else:
                # Empty line
                text_widget.insert(tk.END, '\n', "normal")
        
        i += 1


def create_help_tab(help_frame):
    """Create the help tab with rendered Markdown content."""
    # Create scrollable text widget for help content
    help_text_frame = ttk.Frame(help_frame)
    help_text_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    help_frame.columnconfigure(0, weight=1)
    help_frame.rowconfigure(0, weight=1)
    
    # Create text widget with scrollbar
    help_text = tk.Text(help_text_frame, wrap=tk.WORD, font=("Arial", 10), 
                       state=tk.DISABLED, bg='white', relief=tk.FLAT, 
                       padx=20, pady=20)
    scrollbar = ttk.Scrollbar(help_text_frame, orient=tk.VERTICAL, command=help_text.yview)
    help_text.config(yscrollcommand=scrollbar.set)
    
    help_text.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")
    help_text_frame.columnconfigure(0, weight=1)
    help_text_frame.rowconfigure(0, weight=1)
    
    # Load and display HELP.md content with Markdown rendering
    try:
        with open('HELP.md', 'r', encoding='utf-8') as f:
            help_content = f.read()
        
        help_text.config(state=tk.NORMAL)
        render_markdown_to_text(help_text, help_content)
        help_text.config(state=tk.DISABLED)
        help_text.see(tk.INSERT)  # Scroll to top
    except FileNotFoundError:
        help_text.config(state=tk.NORMAL)
        help_text.insert(tk.END, "Help file (HELP.md) not found in the current directory.", "normal")
        help_text.config(state=tk.DISABLED)
    except Exception as e:
        help_text.config(state=tk.NORMAL)
        help_text.insert(tk.END, f"Error loading help file: {str(e)}", "normal")
        help_text.config(state=tk.DISABLED)


def create_notebook_layout(root):
    """Create the main notebook layout with simulator, output, and help tabs."""
    notebook = ttk.Notebook(root)
    notebook.grid(row=0, column=0, sticky="nsew")
    
    main_frame = ttk.Frame(notebook, padding=12)
    output_frame = ttk.Frame(notebook, padding=12)
    help_frame = ttk.Frame(notebook, padding=12)
    
    # Configure main frame columns
    main_frame.columnconfigure(1, weight=1)
    notebook.add(main_frame, text="Simulator")
    notebook.add(output_frame, text="Detailed Output")
    notebook.add(help_frame, text="Help")
    
    # Set up the help tab
    create_help_tab(help_frame)
    
    return notebook, main_frame, output_frame


def create_polyhedron_selector(main_frame):
    """Create polyhedron dropdown selector."""
    ttk.Label(main_frame, text="Polyhedron").grid(row=0, column=0, sticky="w")
    
    # Create dropdown values with vertex/edge counts
    poly_values = []
    for name, builder in POLYHEDRA.items():
        V, E = builder()
        poly_values.append(f"{name} ({len(V)}V, {len(E)}E)")
    
    poly_combo = ttk.Combobox(main_frame, values=poly_values, state="readonly")
    poly_combo.grid(row=0, column=1, padx=(8, 0), sticky="ew")
    return poly_combo, poly_values


def create_view_controls(main_frame):
    """Create view mode radio buttons."""
    view_var = tk.StringVar(value="Top-Tilt")
    view_frame = ttk.LabelFrame(main_frame, text="View")
    view_frame.grid(row=1, column=0, columnspan=2, pady=(12, 0), sticky="ew")
    for idx, label_text in enumerate(VIEW_MODES.keys()):
        ttk.Radiobutton(
            view_frame,
            text=label_text,
            value=label_text,
            variable=view_var,
        ).grid(row=0, column=idx, padx=8, pady=4)
    return view_var


def create_canvas(main_frame):
    """Create the main drawing canvas."""
    canvas = tk.Canvas(main_frame, width=CANVAS_SIZE, height=CANVAS_SIZE, highlightthickness=0, bg=BACKGROUND)
    canvas.grid(row=2, column=0, columnspan=2, pady=(12, 0))
    return canvas


def create_constraint_controls(main_frame):
    """Create the constraint controls section."""
    controls = ttk.LabelFrame(main_frame, text="Constraints")
    controls.grid(row=3, column=0, columnspan=2, pady=(12, 0), sticky="ew")
    controls.columnconfigure(1, weight=1)
    controls.columnconfigure(2, weight=1)
    controls.columnconfigure(3, weight=1)

    ttk.Label(controls, text="Path length L").grid(row=0, column=0, sticky="w")
    length_var = tk.StringVar(value="2")
    length_entry = ttk.Entry(controls, textvariable=length_var, width=8)
    length_entry.grid(row=0, column=1, sticky="w")

    # Path length mode selection
    path_mode_var = tk.StringVar(value="fixed")
    path_mode_frame = ttk.Frame(controls)
    path_mode_frame.grid(row=0, column=2, columnspan=2, sticky="w", padx=(12,0))
    
    ttk.Radiobutton(path_mode_frame, text="Fixed L", variable=path_mode_var, value="fixed").grid(row=0, column=0, padx=(0,8))
    ttk.Radiobutton(path_mode_frame, text="Variable L", variable=path_mode_var, value="variable").grid(row=0, column=1, padx=(0,8))
    ttk.Radiobutton(path_mode_frame, text="Mixed lengths", variable=path_mode_var, value="mixed").grid(row=0, column=2)

    ttk.Label(controls, text="Sampling iterations").grid(row=1, column=0, sticky="w", pady=(6,0))
    iter_var = tk.StringVar(value="2000")
    iter_entry = ttk.Entry(controls, textvariable=iter_var, width=12)
    iter_entry.grid(row=1, column=1, sticky="w", pady=(6,0))

    ttk.Label(controls, text="Fixed endpoints (0=minimize)").grid(row=1, column=2, sticky="w", padx=(12,0), pady=(6,0))
    endpoints_var = tk.StringVar(value="0")
    endpoints_entry = ttk.Entry(controls, textvariable=endpoints_var, width=8)
    endpoints_entry.grid(row=1, column=3, sticky="w", pady=(6,0))

    dc_only_var = tk.BooleanVar(value=False)
    dc_only_check = ttk.Checkbutton(controls, text="DC only (no alternating vertices)", variable=dc_only_var)
    dc_only_check.grid(row=2, column=0, sticky="w", pady=(6,0))

    alternating_only_var = tk.BooleanVar(value=False)
    alternating_only_check = ttk.Checkbutton(controls, text="Alternating only (all vertices are alternating)", variable=alternating_only_var)
    alternating_only_check.grid(row=2, column=1, sticky="w", pady=(6,0))

    sneak_free_var = tk.BooleanVar(value=False)
    sneak_free_check = ttk.Checkbutton(controls, text="Sneak path free (requires DC only)", variable=sneak_free_var, state="disabled")
    sneak_free_check.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6,0))

    bipolar_only_var = tk.BooleanVar(value=False)
    bipolar_only_check = ttk.Checkbutton(controls, text="Bipolar driving scheme (no tristate Z needed)", variable=bipolar_only_var)
    bipolar_only_check.grid(row=3, column=1, columnspan=2, sticky="w", pady=(6,0))

    equal_current_var = tk.BooleanVar(value=False)
    equal_current_check = ttk.Checkbutton(controls, text="Equal current (all edges carry equal current)", variable=equal_current_var)
    equal_current_check.grid(row=4, column=0, columnspan=2, sticky="w", pady=(6,0))

    # Constraint interdependencies
    def on_dc_only_change():
        """Enable/disable sneak path free checkbox based on DC only selection."""
        if dc_only_var.get():
            sneak_free_check.config(state="normal")
            # Disable alternating only when DC only is selected (mutually exclusive)
            alternating_only_var.set(False)
            alternating_only_check.config(state="disabled")
        else:
            sneak_free_var.set(False)  # Uncheck when disabled
            sneak_free_check.config(state="disabled")
            alternating_only_check.config(state="normal")
    
    def on_alternating_only_change():
        """Handle alternating only constraint changes."""
        if alternating_only_var.get():
            # Disable DC only when alternating only is selected (mutually exclusive)
            dc_only_var.set(False)
            dc_only_check.config(state="disabled")
            sneak_free_var.set(False)
            sneak_free_check.config(state="disabled")
        else:
            dc_only_check.config(state="normal")
    
    def on_path_mode_change():
        """Handle path mode radio button changes."""
        mode = path_mode_var.get()
        if mode == "fixed":
            # Fixed path length: enable path length input and fixed endpoints
            length_entry.config(state="normal")
            endpoints_entry.config(state="normal")
        elif mode in ["variable", "mixed"]:
            # Variable/Mixed path length: disable path length input and fixed endpoints
            length_entry.config(state="disabled")
            endpoints_entry.config(state="disabled")
            endpoints_var.set("0")
    
    dc_only_var.trace_add("write", lambda *args: on_dc_only_change())
    alternating_only_var.trace_add("write", lambda *args: on_alternating_only_change())
    path_mode_var.trace_add("write", lambda *args: on_path_mode_change())
    
    return {
        'controls': controls,
        'length_var': length_var,
        'iter_var': iter_var,
        'endpoints_var': endpoints_var,
        'path_mode_var': path_mode_var,
        'dc_only_var': dc_only_var,
        'alternating_only_var': alternating_only_var,
        'sneak_free_var': sneak_free_var,
        'bipolar_only_var': bipolar_only_var,
        'equal_current_var': equal_current_var
    }


def create_status_and_buttons(main_frame, controls):
    """Create status display and control buttons."""
    status_var = tk.StringVar(value="Select parameters and run the solver.")
    status_label = ttk.Label(main_frame, textvariable=status_var)
    status_label.grid(row=4, column=0, columnspan=2, pady=(10,0), sticky="w")

    progress = ttk.Progressbar(main_frame, mode="determinate", maximum=100, length=240)
    progress.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(4,0))

    # Button frame for Run and Stop buttons
    button_frame = ttk.Frame(controls)
    button_frame.grid(row=5, column=0, columnspan=4, pady=(12,0))
    
    run_button = ttk.Button(button_frame, text="Run Solver")
    run_button.grid(row=0, column=0, padx=(0, 8))
    
    # Cancellation event for stopping the solver
    cancel_event = threading.Event()
    
    stop_button = ttk.Button(button_frame, text="Stop", state="disabled")
    stop_button.grid(row=0, column=1)
    
    return {
        'status_var': status_var,
        'progress': progress,
        'run_button': run_button,
        'stop_button': stop_button,
        'cancel_event': cancel_event
    }


def main():
    root = create_main_window()
    current_highlight = {}

    notebook, main_frame, output_frame = create_notebook_layout(root)
    poly_combo, poly_values = create_polyhedron_selector(main_frame)
    view_var = create_view_controls(main_frame)
    canvas = create_canvas(main_frame)
    constraint_vars = create_constraint_controls(main_frame)
    controls = constraint_vars['controls']

    # Extract variables from the constraint controls  
    length_var = constraint_vars['length_var']
    iter_var = constraint_vars['iter_var']
    endpoints_var = constraint_vars['endpoints_var']
    path_mode_var = constraint_vars['path_mode_var']
    dc_only_var = constraint_vars['dc_only_var']
    alternating_only_var = constraint_vars['alternating_only_var']
    sneak_free_var = constraint_vars['sneak_free_var']
    bipolar_only_var = constraint_vars['bipolar_only_var']
    equal_current_var = constraint_vars['equal_current_var']

    ui_controls = create_status_and_buttons(main_frame, controls)
    status_var = ui_controls['status_var']
    progress = ui_controls['progress']
    run_button = ui_controls['run_button']
    stop_button = ui_controls['stop_button']
    cancel_event = ui_controls['cancel_event']

    # State management
    progress_job = [None]
    current_thread = [None]

    summary_var = tk.StringVar(value="")
    summary_label = ttk.Label(main_frame, textvariable=summary_var, font=("Helvetica", 11, "bold"))
    summary_label.grid(row=6, column=0, columnspan=2, pady=(10,0), sticky="w")

    output_text = tk.Text(output_frame, wrap="word", height=30)
    output_text.pack(fill="both", expand=True)
    output_text.configure(state="disabled")

    def get_polyhedron_name():
        """Extract the polyhedron name from the combo box selection."""
        selection = poly_combo.get()
        if not selection:
            return None
        # Extract name before the parentheses
        return selection.split(' (')[0]
    
    def update_canvas(*_):
        name = get_polyhedron_name()
        builder = POLYHEDRA.get(name)
        if builder is None:
            canvas.delete("all")
            return
        active_view = VIEW_MODES.get(view_var.get(), (ISO_Y, ISO_X))
        highlight = current_highlight if current_highlight.get('edges') else {}
        current_data = current_highlight.get('current_data', None)
        draw_polyhedron(canvas, builder, active_view, highlight=highlight, current_data=current_data)

    def on_poly_change(event=None):
        current_highlight.clear()
        update_canvas()

    def on_view_change(*_):
        update_canvas()

    poly_combo.bind("<<ComboboxSelected>>", on_poly_change)
    view_var.trace_add("write", on_view_change)

    poly_combo.set(poly_values[0])
    update_canvas()

    def stop_solver():
        """Stop the currently running solver"""
        cancel_event.set()  # Signal the worker thread to stop
        if current_thread[0] and current_thread[0].is_alive():
            status_var.set("Stopping solver...")
        else:
            set_running(False)
            status_var.set("Solver stopped.")

    # Button frame for Run and Stop buttons
    button_frame = ttk.Frame(controls)
    button_frame.grid(row=5, column=0, columnspan=4, pady=(12,0))
    
    run_button = ttk.Button(button_frame, text="Run Solver")
    run_button.grid(row=0, column=0, padx=(0, 8))
    
    stop_button = ttk.Button(button_frame, text="Stop", state="disabled", command=stop_solver)
    stop_button.grid(row=0, column=1)

    def stop_progress():
        job = progress_job[0]
        if job is not None:
            root.after_cancel(job)
            progress_job[0] = None
        progress['value'] = 0

    def start_progress():
        stop_progress()

        def pulse():
            progress['value'] = (progress['value'] + 5) % 100
            progress_job[0] = root.after(80, pulse)

        pulse()

    def set_running(running: bool):
        if running:
            run_button.config(state="disabled")
            stop_button.config(state="normal")
            start_progress()
            status_var.set("Running solver...")
            cancel_event.clear()  # Reset cancellation flag
        else:
            run_button.config(state="normal")
            stop_button.config(state="disabled") 
            stop_progress()

    def launch():
        # Only validate L_val if using fixed path mode
        path_mode = path_mode_var.get()
        if path_mode == "fixed":
            try:
                L_val = int(length_var.get())
                if L_val <= 0:
                    raise ValueError
            except ValueError:
                status_var.set("L must be a positive integer.")
                return
        else:
            L_val = 2  # Default value for variable/mixed path modes (not actually used)
        try:
            iter_val = int(iter_var.get())
            if iter_val <= 0:
                raise ValueError
        except ValueError:
            status_var.set("Iterations must be a positive integer.")
            return

        name = get_polyhedron_name()
        builder = POLYHEDRA.get(name)
        if builder is None:
            status_var.set("Please choose a polyhedron.")
            return

        current_highlight.clear()
        update_canvas()
        set_running(True)

        def worker():
            try:
                # Check for cancellation before starting
                if cancel_event.is_set():
                    def cancelled():
                        set_running(False)
                        status_var.set("Solver cancelled.")
                    root.after(0, cancelled)
                    return
                
                dc_only_requested = dc_only_var.get()
                sneak_free_requested = sneak_free_var.get()
                equal_current_requested = equal_current_var.get()
                alternating_only_requested = alternating_only_var.get()
                bipolar_only_requested = bipolar_only_var.get()
                path_mode_requested = path_mode_var.get()
                
                # Get fixed endpoints value
                try:
                    fixed_endpoints_val = int(endpoints_var.get())
                    if fixed_endpoints_val < 0:
                        fixed_endpoints_val = 0
                except ValueError:
                    fixed_endpoints_val = 0
                
                # Choose solver based on path mode setting
                if path_mode_requested == "fixed":
                    # Fixed path length: original behavior
                    result = run_solver(name, builder, L_val, iter_val, cancel_event, dc_only=dc_only_requested, sneak_free=sneak_free_requested, equal_current=equal_current_requested, alternating_only=alternating_only_requested, bipolar_only=bipolar_only_requested, fixed_endpoints=fixed_endpoints_val)
                elif path_mode_requested == "variable":
                    # Variable path length: try different L values, all paths in solution have same L
                    result = run_variable_path_solver(name, builder, iter_val, cancel_event, dc_only=dc_only_requested, sneak_free=sneak_free_requested, equal_current=equal_current_requested, alternating_only=alternating_only_requested, bipolar_only=bipolar_only_requested)
                elif path_mode_requested == "mixed":
                    # Mixed path length: allow paths of different lengths within same solution
                    result = run_mixed_path_solver(name, builder, iter_val, cancel_event, dc_only=dc_only_requested, sneak_free=sneak_free_requested, equal_current=equal_current_requested, alternating_only=alternating_only_requested, bipolar_only=bipolar_only_requested)
                
                # Check if solver was cancelled with no results
                if result is None:  # Cancelled with no solution found
                    def cancelled():
                        set_running(False)
                        status_var.set("Solver cancelled - no solution found.")
                        summary_var.set("Operation cancelled - no solution found.")
                    root.after(0, cancelled)
                    return
                
                res, mode, duration, total_edges = result
                
                # For variable/mixed path modes, determine the actual path length used
                if path_mode_requested in ["variable", "mixed"] and res is not None:
                    # Get the path length from the solution
                    chosen_paths = res[5]
                    if path_mode_requested == "mixed":
                        # For mixed mode, show "Mixed" as the length since paths can have different lengths
                        display_L = "Mixed"
                    else:
                        # For variable mode, show the actual uniform length found
                        actual_L = len(chosen_paths[0]) - 1 if chosen_paths else L_val
                        display_L = actual_L
                    fixed_endpoints_val = 0  # Variable/mixed path modes don't use fixed endpoints
                else:
                    display_L = L_val
                
                detail = format_result(name, display_L, res, total_edges=total_edges, dc_only=dc_only_requested, sneak_free=sneak_free_requested, equal_current=equal_current_requested, alternating_only=alternating_only_requested, bipolar_only=bipolar_only_requested, fixed_endpoints=fixed_endpoints_val, path_mode=path_mode_requested)
                if res is None:
                    summary = "No solution found."
                    highlight_payload = {}
                else:
                    # Check for equal current coverage
                    n_end, n_paths, endpoints, counts, dir_edges, chosen_paths = res
                    flow_data = analyze_current_flow(chosen_paths, dir_edges)
                    currents = [data['current'] for data in flow_data.values()]
                    equal_current = len(set(f"{c:.6f}" for c in currents)) == 1 if currents else False
                    
                    # Calculate coverage for display
                    covered_edges = set()
                    for path in chosen_paths:
                        for i in range(len(path) - 1):
                            covered_edges.add((path[i], path[i + 1]))
                    coverage_edges = len(covered_edges)
                    
                    # Build coverage ratio text
                    if coverage_edges < total_edges:
                        coverage_ratio = coverage_edges / total_edges
                        coverage_text = f", coverage={coverage_ratio:.3f} ({coverage_edges}/{total_edges})"
                    else:
                        coverage_text = ""
                    
                    # Add partial result indicator
                    partial_indicator = " [PARTIAL]" if "STOPPED" in mode else ""
                    
                    if equal_current:
                        uniform_current = currents[0] if currents else 0
                        summary = f"Solution found: endpoints={res[0]}, paths={res[1]}{coverage_text} (Equal current: {uniform_current:.3f}){partial_indicator}"
                    else:
                        summary = f"Solution found: endpoints={res[0]}, paths={res[1]}{coverage_text}{partial_indicator}"
                    
                    oriented_edges = []
                    seen = set()
                    for path in chosen_paths:
                        for edge in zip(path, path[1:]):
                            key = tuple(sorted(edge))
                            if key not in seen:
                                seen.add(key)
                                oriented_edges.append(edge)
                    highlight_payload = {'endpoints': endpoints, 'edges': oriented_edges, 'current_data': flow_data, 'all_directed_edges': dir_edges}
                def finish():
                    set_running(False)
                    # Check if this was a partial result from stopping
                    if "STOPPED" in mode:
                        status_var.set(f"Stopped early - showing best result found using {mode} in {duration:.2f}s")
                    else:
                        status_var.set(f"Completed using {mode} in {duration:.2f}s")
                    summary_var.set(summary)
                    current_highlight.clear()
                    current_highlight.update(highlight_payload)
                    update_canvas()
                    output_text.configure(state="normal")
                    output_text.delete("1.0", tk.END)
                    output_text.insert(tk.END, detail)
                    output_text.configure(state="disabled")
                root.after(0, finish)
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)
                def fail():
                    set_running(False)
                    status_var.set(f"Error: {error_msg}")
                root.after(0, fail)

        thread = threading.Thread(target=worker, daemon=True)
        current_thread[0] = thread
        thread.start()

    run_button.config(command=launch)

    root.mainloop()


if __name__ == "__main__":
    main()
