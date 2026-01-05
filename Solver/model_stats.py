#!/usr/bin/env python3
"""
Scan a folder of 3D object files and report per-object stats plus regularity totals.

Supported formats: .json (vertices/edges/faces), .obj, .off
"""

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

SUPPORTED_EXTS = {".json", ".obj", ".off"}


def parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_edge(u, v):
    if u is None or v is None or u == v:
        return None
    return (u, v) if u < v else (v, u)


def edges_from_faces(faces):
    edges = set()
    for face in faces:
        if not isinstance(face, (list, tuple)) or len(face) < 2:
            continue
        indices = [parse_int(v) for v in face]
        if any(v is None for v in indices):
            continue
        n = len(indices)
        for i in range(n):
            edge = normalize_edge(indices[i], indices[(i + 1) % n])
            if edge:
                edges.add(edge)
    return edges


def parse_edges_from_json(edges_raw):
    edges = []
    for e in edges_raw:
        u = v = None
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            u = parse_int(e[0])
            v = parse_int(e[1])
        elif isinstance(e, dict):
            if "source" in e and "target" in e:
                u = parse_int(e.get("source"))
                v = parse_int(e.get("target"))
            elif "u" in e and "v" in e:
                u = parse_int(e.get("u"))
                v = parse_int(e.get("v"))
            elif "from" in e and "to" in e:
                u = parse_int(e.get("from"))
                v = parse_int(e.get("to"))
        if u is None or v is None:
            continue
        edges.append((u, v))
    return edges


def load_json_model(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vertices = data.get("vertices")
    n_vertices = len(vertices) if isinstance(vertices, list) else None

    edges_raw = data.get("edges") or []
    edges = parse_edges_from_json(edges_raw) if edges_raw else []
    if not edges:
        faces = data.get("faces") or []
        edges = list(edges_from_faces(faces))

    if n_vertices is None and edges:
        n_vertices = max(max(u, v) for u, v in edges) + 1

    if n_vertices is None:
        raise ValueError("missing vertices and edges")

    edges_set = set()
    for u, v in edges:
        edge = normalize_edge(u, v)
        if edge:
            edges_set.add(edge)

    return n_vertices, edges_set, data.get("name")


def parse_obj_index(token, n_vertices):
    if not token:
        return None
    val = parse_int(token)
    if val is None:
        return None
    if val < 0:
        return n_vertices + val
    return val - 1


def load_obj_model(path):
    n_vertices = 0
    edges = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                n_vertices += 1
                continue
            if line.startswith("f ") or line.startswith("l "):
                parts = line.split()[1:]
                indices = []
                for part in parts:
                    token = part.split("/")[0]
                    idx = parse_obj_index(token, n_vertices)
                    if idx is None:
                        continue
                    indices.append(idx)
                if len(indices) < 2:
                    continue
                is_face = line.startswith("f ")
                limit = len(indices) if is_face else len(indices) - 1
                for i in range(limit):
                    a = indices[i]
                    b = indices[(i + 1) % len(indices)] if is_face else indices[i + 1]
                    edge = normalize_edge(a, b)
                    if edge:
                        edges.add(edge)

    if n_vertices == 0:
        raise ValueError("no vertices found")

    return n_vertices, edges, None


def next_data_line(lines, start_idx):
    idx = start_idx
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith("#"):
            continue
        return line, idx
    return None, idx


def load_off_model(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    line, idx = next_data_line(lines, 0)
    if line is None:
        raise ValueError("empty file")

    parts = line.split()
    if parts[0] in ("OFF", "COFF"):
        line, idx = next_data_line(lines, idx)
        if line is None:
            raise ValueError("missing counts")
        parts = line.split()
    elif parts[0].startswith("OFF"):
        parts = parts[1:]
    else:
        raise ValueError("missing OFF header")

    if len(parts) < 2:
        raise ValueError("invalid counts line")
    n_vertices = parse_int(parts[0])
    n_faces = parse_int(parts[1])
    if n_vertices is None or n_faces is None:
        raise ValueError("invalid counts")

    for _ in range(n_vertices):
        line, idx = next_data_line(lines, idx)
        if line is None:
            raise ValueError("unexpected EOF while reading vertices")

    edges = set()
    for _ in range(n_faces):
        line, idx = next_data_line(lines, idx)
        if line is None:
            break
        parts = line.split()
        if not parts:
            continue
        count = parse_int(parts[0])
        if count is None or count < 2:
            continue
        if len(parts) < count + 1:
            continue
        indices = [parse_int(v) for v in parts[1:count + 1]]
        if any(v is None for v in indices):
            continue
        for i in range(count):
            edge = normalize_edge(indices[i], indices[(i + 1) % count])
            if edge:
                edges.add(edge)

    return n_vertices, edges, None


def load_model(path):
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json_model(path)
    if suffix == ".obj":
        return load_obj_model(path)
    if suffix == ".off":
        return load_off_model(path)
    raise ValueError(f"unsupported extension: {suffix}")


def format_degrees(degrees):
    if not degrees:
        return "-"
    return ", ".join(str(d) for d in sorted(degrees))


def format_degree_counts(degrees):
    if not degrees:
        return "-"
    counts = Counter(degrees)
    return ", ".join(f"{deg}:{counts[deg]}" for deg in sorted(counts))


def compute_stats(n_vertices, edges):
    if n_vertices <= 0:
        return n_vertices, [], set()
    if edges:
        max_idx = max(max(u, v) for u, v in edges)
        if max_idx >= n_vertices:
            n_vertices = max_idx + 1
    degrees = [0] * n_vertices
    for u, v in edges:
        if u < 0 or v < 0:
            continue
        if u >= len(degrees) or v >= len(degrees):
            continue
        degrees[u] += 1
        degrees[v] += 1
    return n_vertices, degrees, set(degrees)


def filtered_edges(n_vertices, edges):
    return [(u, v) for u, v in edges if 0 <= u < n_vertices and 0 <= v < n_vertices]


def is_bipartite(n_vertices, edges):
    if n_vertices <= 0:
        return None
    adj = [[] for _ in range(n_vertices)]
    for u, v in filtered_edges(n_vertices, edges):
        adj[u].append(v)
        adj[v].append(u)
    color = [None] * n_vertices
    for start in range(n_vertices):
        if color[start] is not None:
            continue
        color[start] = 0
        queue = [start]
        while queue:
            u = queue.pop()
            for v in adj[u]:
                if color[v] is None:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False
    return True


def is_cycle_decomposition_possible(n_vertices, edges):
    if n_vertices <= 0:
        return None
    degree = [0] * n_vertices
    for u, v in filtered_edges(n_vertices, edges):
        degree[u] += 1
        degree[v] += 1
    return all(deg % 2 == 0 for deg in degree)


def all_pairs_shortest_paths(n_vertices, edges):
    if n_vertices <= 0:
        return None
    adj = [[] for _ in range(n_vertices)]
    for u, v in filtered_edges(n_vertices, edges):
        adj[u].append(v)
        adj[v].append(u)
    dist_matrix = []
    for start in range(n_vertices):
        dist = [-1] * n_vertices
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


def graph_diameter_from_dist(dist_matrix):
    if dist_matrix is None:
        return None
    if not dist_matrix:
        return 0
    return max(max(row) for row in dist_matrix)


def graph_diameter(n_vertices, edges):
    dist_matrix = all_pairs_shortest_paths(n_vertices, edges)
    return graph_diameter_from_dist(dist_matrix)


def build_adj(n_vertices, edges):
    adj = [set() for _ in range(n_vertices)]
    for u, v in filtered_edges(n_vertices, edges):
        adj[u].add(v)
        adj[v].add(u)
    return adj


def enumerate_cycles(adj, length, max_cycles=50000):
    n = len(adj)
    found = set()

    for start in range(n):
        if len(found) >= max_cycles:
            break
        stack = [(start, [start], {start})]
        while stack:
            if len(found) >= max_cycles:
                break
            v, path, visited = stack.pop()
            if len(path) == length:
                if start in adj[v]:
                    cyc = tuple(path)
                    rev = tuple(reversed(path))
                    rotations = []
                    for i in range(length):
                        rotations.append(cyc[i:] + cyc[:i])
                        rotations.append(rev[i:] + rev[:i])
                    found.add(min(rotations))
                continue
            if len(path) >= length:
                continue
            for u in adj[v]:
                if u == start and len(path) >= 3:
                    continue
                if u < start:
                    continue
                if u in visited:
                    continue
                stack.append((u, path + [u], visited | {u}))

    return [list(c) for c in sorted(found)]


def edges_of_cycle(cycle):
    k = len(cycle)
    return [
        normalize_edge(cycle[i], cycle[(i + 1) % k])
        for i in range(k)
    ]


def cycle_opposite_pairs_ok(cycle, dist_matrix, diameter):
    if dist_matrix is None:
        return False
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


def solve_exact_cover(all_edges, cycles, cycle_edges):
    edge_to_cycles = {e: [] for e in all_edges}
    for ci, es in enumerate(cycle_edges):
        for e in es:
            if e in edge_to_cycles:
                edge_to_cycles[e].append(ci)

    for e, cands in edge_to_cycles.items():
        if not cands:
            return None

    uncovered = set(all_edges)
    chosen = []
    cycle_edge_sets = cycle_edges

    def pick_edge():
        best_e = None
        best_count = None
        for e in uncovered:
            count = sum(
                1 for ci in edge_to_cycles[e]
                if cycle_edge_sets[ci].issubset(uncovered)
            )
            if best_count is None or count < best_count:
                best_count = count
                best_e = e
                if count <= 1:
                    break
        return best_e

    def backtrack():
        if not uncovered:
            return True
        e = pick_edge()
        if e is None:
            return False
        for ci in edge_to_cycles[e]:
            ces = cycle_edge_sets[ci]
            if not ces.issubset(uncovered):
                continue
            chosen.append(ci)
            uncovered.difference_update(ces)
            if backtrack():
                return True
            uncovered.update(ces)
            chosen.pop()
        return False

    if backtrack():
        return chosen
    return None


def cycle_decomposition_length_twice_diameter(n_vertices, edges, diameter, dist_matrix):
    if diameter is None:
        return False
    if not edges:
        return True
    if n_vertices <= 0:
        return False
    length = 2 * diameter
    if length < 4:
        return False
    if len(edges) % length != 0:
        return False

    degrees = [0] * n_vertices
    for u, v in filtered_edges(n_vertices, edges):
        degrees[u] += 1
        degrees[v] += 1
    if any(deg % 2 != 0 for deg in degrees):
        return False

    adj = build_adj(n_vertices, edges)
    cycles = enumerate_cycles(adj, length)
    if not cycles:
        return False

    all_edges_set = set(edges)
    valid_cycles = []
    valid_edges = []
    for cycle in cycles:
        if not cycle_opposite_pairs_ok(cycle, dist_matrix, diameter):
            continue
        es = set(edges_of_cycle(cycle))
        if None in es or not es.issubset(all_edges_set):
            continue
        valid_cycles.append(cycle)
        valid_edges.append(es)

    if not valid_cycles:
        return False

    solution = solve_exact_cover(all_edges_set, valid_cycles, valid_edges)
    return solution is not None


def print_table(rows):
    headers = [
        "Name",
        "V",
        "E",
        "Degrees",
        "Diameter",
        "DegreeCounts",
        "Regularity",
        "Bipartite",
        "CycleDecomp",
        "CycleDecomp2D",
    ]
    print("\t".join(headers))
    for row in rows:
        print("\t".join([
            str(row["name"]),
            str(row["vertices"]),
            str(row["edges"]),
            str(row["degrees"]),
            str(row["diameter"]),
            str(row["degree_counts"]),
            str(row["regularity"]),
            str(row["bipartite"]),
            str(row["cycle_decomp"]),
            str(row["cycle_decomp_2d"]),
        ]))


def main():
    parser = argparse.ArgumentParser(
        description="Report vertex/edge statistics for 3D object files in a folder."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing object files (e.g. Solver/models_test)",
    )
    args = parser.parse_args()

    folder = args.folder
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        return 1

    rows = []
    errors = []
    regular_counts = defaultdict(int)
    mixed_count = 0

    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        try:
            n_vertices, edges, name = load_model(path)
            n_vertices, degrees, degree_set = compute_stats(n_vertices, edges)
            if n_vertices is None or n_vertices <= 0:
                raise ValueError("invalid vertex count")
            regularity = "mixed"
            if degree_set:
                if len(degree_set) == 1:
                    degree = next(iter(degree_set))
                    regularity = f"{degree}-regular"
                    regular_counts[degree] += 1
                else:
                    mixed_count += 1
            else:
                regularity = "invalid"

            display_name = name if name else path.stem
            bipartite = is_bipartite(n_vertices, edges)
            bipartite_str = "yes" if bipartite else "no"
            cycle_decomp = is_cycle_decomposition_possible(n_vertices, edges)
            cycle_decomp_str = "yes" if cycle_decomp else "no"
            dist_matrix = all_pairs_shortest_paths(n_vertices, edges)
            diameter = graph_diameter_from_dist(dist_matrix)
            diameter_str = "-" if diameter is None else str(diameter)
            cycle_decomp_2d = cycle_decomposition_length_twice_diameter(
                n_vertices, edges, diameter, dist_matrix
            )
            cycle_decomp_2d_str = "yes" if cycle_decomp_2d else "no"
            rows.append({
                "name": display_name,
                "vertices": n_vertices,
                "edges": len(edges),
                "degrees": format_degrees(degree_set),
                "degree_counts": format_degree_counts(degrees),
                "regularity": regularity,
                "bipartite": bipartite_str,
                "cycle_decomp": cycle_decomp_str,
                "diameter": diameter_str,
                "cycle_decomp_2d": cycle_decomp_2d_str,
            })
        except Exception as exc:
            errors.append((path.name, str(exc)))

    if rows:
        rows.sort(key=lambda r: r["name"].lower())
        print_table(rows)
    else:
        print("Name\tV\tE\tDegrees\tDiameter\tDegreeCounts\tRegularity\tBipartite\tCycleDecomp\tCycleDecomp2D")

    total_objects = len(rows)
    regular_total = sum(regular_counts.values())
    other_regular = {
        degree: count for degree, count in regular_counts.items() if degree not in (3, 4)
    }

    print("\nTotals\tValue")
    print(f"Total objects\t{total_objects}")
    print(f"3-regular\t{regular_counts.get(3, 0)}")
    print(f"4-regular\t{regular_counts.get(4, 0)}")
    if other_regular:
        other_str = ", ".join(f"{deg}-regular:{cnt}" for deg, cnt in sorted(other_regular.items()))
        print(f"n-regular\t{sum(other_regular.values())} ({other_str})")
    else:
        print("n-regular\t0")
    print(f"Mixed\t{mixed_count}")
    print(f"Regular total\t{regular_total}")

    if errors:
        print("\nSkippedFile\tReason")
        for name, msg in errors:
            print(f"{name}\t{msg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
