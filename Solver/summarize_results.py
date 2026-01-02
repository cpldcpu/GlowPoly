#!/usr/bin/env python3
"""
Summarize solver output JSON files.
Shows: name, vertices, edges, vertex degrees, path length, and number of taps.
Grouped by solution type.
"""

import argparse
import json
from pathlib import Path

def format_degrees(degrees):
    """Format vertex degrees as a string."""
    if not degrees:
        return "-"
    return ", ".join(str(d) for d in sorted(degrees))

def load_json(filepath):
    """Load JSON file and return data."""
    with open(filepath, 'r') as f:
        return json.load(f)

def calc_vertex_degrees(edges, n_vertices):
    """Calculate unique vertex degrees from edge list."""
    degree_count = [0] * n_vertices
    for e in edges:
        degree_count[e[0]] += 1
        degree_count[e[1]] += 1
    return sorted(set(degree_count))

def load_model_degrees(models_dir):
    """Load all models and return a dict mapping name -> vertex degrees."""
    degrees_map = {}
    if not models_dir.exists():
        return degrees_map
    for model_file in models_dir.glob("*.json"):
        try:
            data = load_json(model_file)
            edges = data.get("edges", [])
            n_vertices = len(data.get("vertices", []))
            if edges and n_vertices:
                degrees = calc_vertex_degrees(edges, n_vertices)
                # Map by filename (with .json) and by name field
                degrees_map[model_file.name] = degrees
                if "name" in data:
                    degrees_map[data["name"]] = degrees
        except Exception:
            pass
    return degrees_map

def summarize_geodesic(data, degrees_map=None):
    """Summarize geodesic cover results."""
    if degrees_map is None:
        degrees_map = {}
    results = []
    for item in data:
        name = item.get("name", "unknown")
        nV = item.get("nV", 0)
        nE = item.get("nE", 0)
        # Look up degrees from model files
        degrees_list = degrees_map.get(name)
        degrees = format_degrees(degrees_list) if degrees_list else "-"
        path_length = item.get("best_L", "-")
        # Count unique endpoint vertices (taps) from solution
        solution = item.get("solution", [])
        if solution:
            endpoints = set()
            for pair in solution:
                endpoints.add(pair.get("s"))
                endpoints.add(pair.get("t"))
            num_taps = len(endpoints)
        else:
            num_taps = "-"
        results.append({
            "name": name,
            "vertices": nV,
            "edges": nE,
            "degrees": degrees,
            "path_length": path_length,
            "taps": num_taps
        })
    return results

def summarize_bidirectional(data):
    """Summarize bidirectional path results."""
    results = []
    for item in data:
        if not item.get("ok", False):
            continue
        name = item.get("model", "unknown").replace(".json", "")
        nV = item.get("n_vertices", 0)
        nE = item.get("n_edges", 0)
        degrees = format_degrees(item.get("vertex_degrees", []))
        path_length = item.get("path_length", "-")
        # Taps = number of endpoint vertices (always 2 for bidirectional)
        endpoint_pair = item.get("endpoint_pair", [])
        num_taps = len(endpoint_pair) if endpoint_pair else "-"
        results.append({
            "name": name,
            "vertices": nV,
            "edges": nE,
            "degrees": degrees,
            "path_length": path_length,
            "taps": num_taps
        })
    return results

def summarize_cycle_decomp(data):
    """Summarize cycle decomposition results."""
    results = []
    for item in data:
        if not item.get("ok", False):
            continue
        name = item.get("model", "unknown").replace(".json", "")
        nV = item.get("n_vertices", 0)
        nE = item.get("n_edges", 0)
        degrees = format_degrees(item.get("vertex_degrees", []))
        cycle_length = item.get("cycle_length", "-")
        num_taps = item.get("num_unique_endpoints", "-")
        results.append({
            "name": name,
            "vertices": nV,
            "edges": nE,
            "degrees": degrees,
            "path_length": cycle_length,
            "taps": num_taps
        })
    return results

def safe_str(val):
    """Convert value to string, handling None."""
    if val is None:
        return "-"
    return str(val)

def filter_by_taps(results, max_taps):
    """Filter results to only include those with taps <= max_taps."""
    if max_taps is None:
        return results
    filtered = []
    for r in results:
        taps = r.get("taps")
        if taps is not None and taps != "-":
            try:
                if int(taps) <= max_taps:
                    filtered.append(r)
            except (ValueError, TypeError):
                pass
    return filtered

def print_table(title, results, max_taps=None):
    """Print a formatted table of results."""
    # Apply filter if specified
    if max_taps is not None:
        results = filter_by_taps(results, max_taps)
        title = f"{title} [taps <= {max_taps}]"

    if not results:
        print(f"\n{title}")
        print("-" * len(title))
        print("  No matching results.\n")
        return

    # Calculate column widths
    headers = ["Name", "V", "E", "Degrees", "Path Len", "Taps"]
    widths = [
        max(len(headers[0]), max(len(safe_str(r["name"])) for r in results)),
        max(len(headers[1]), max(len(safe_str(r["vertices"])) for r in results)),
        max(len(headers[2]), max(len(safe_str(r["edges"])) for r in results)),
        max(len(headers[3]), max(len(safe_str(r["degrees"])) for r in results)),
        max(len(headers[4]), max(len(safe_str(r["path_length"])) for r in results)),
        max(len(headers[5]), max(len(safe_str(r["taps"])) for r in results)),
    ]

    # Print header
    print(f"\n{title}")
    print("=" * len(title))
    header_fmt = f"  {{:<{widths[0]}}}  {{:>{widths[1]}}}  {{:>{widths[2]}}}  {{:<{widths[3]}}}  {{:>{widths[4]}}}  {{:>{widths[5]}}}"
    print(header_fmt.format(*headers))
    print("  " + "-" * (sum(widths) + 10))

    # Print rows
    for r in sorted(results, key=lambda x: x["name"]):
        print(header_fmt.format(
            safe_str(r["name"]),
            safe_str(r["vertices"]),
            safe_str(r["edges"]),
            safe_str(r["degrees"]),
            safe_str(r["path_length"]),
            safe_str(r["taps"])
        ))
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Summarize solver output JSON files."
    )
    parser.add_argument(
        "--max-taps", "-t",
        type=int,
        metavar="N",
        help="Only show results with at most N taps"
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent / "output"
    models_dir = Path(__file__).parent / "models"

    # Load vertex degrees from model files
    degrees_map = load_model_degrees(models_dir)

    print("\n" + "=" * 60)
    print("SOLVER RESULTS SUMMARY")
    if args.max_taps:
        print(f"  Filter: taps <= {args.max_taps}")
    print("=" * 60)

    # Geodesic Cover Results
    geodesic_file = output_dir / "geodesic_cover_results.json"
    if geodesic_file.exists():
        data = load_json(geodesic_file)
        results = summarize_geodesic(data, degrees_map)
        print_table(f"Geodesic Cover ({len(results)} models)", results, args.max_taps)

    # Bidirectional Path Results
    bidirectional_file = output_dir / "bidirectional_path_results.json"
    if bidirectional_file.exists():
        data = load_json(bidirectional_file)
        results = summarize_bidirectional(data)
        print_table(f"Bidirectional Path ({len(results)} solutions)", results, args.max_taps)

    # Cycle Decomposition Results
    cycle_file = output_dir / "cycle_decomp_results.json"
    if cycle_file.exists():
        data = load_json(cycle_file)
        results = summarize_cycle_decomp(data)
        print_table(f"Cycle Decomposition ({len(results)} solutions)", results, args.max_taps)

    # Summary stats
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if bidirectional_file.exists():
        bidir_data = load_json(bidirectional_file)
        total = len(bidir_data)
        ok = sum(1 for item in bidir_data if item.get("ok", False))
        print(f"  Bidirectional:      {ok:3d} / {total:3d} models have solutions")

    if cycle_file.exists():
        cycle_data = load_json(cycle_file)
        total = len(cycle_data)
        ok = sum(1 for item in cycle_data if item.get("ok", False))
        print(f"  Cycle Decomp:       {ok:3d} / {total:3d} models have solutions")

    if geodesic_file.exists():
        geo_data = load_json(geodesic_file)
        print(f"  Geodesic Cover:     {len(geo_data):3d} models processed")

    print()

if __name__ == "__main__":
    main()
