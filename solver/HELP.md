# Glowing Polyhedron Simulator - Help

## Overview

This interactive tool optimizes LED path planning on 3D polyhedra. It finds optimal paths that minimize endpoints while maximizing edge coverage, subject to various electrical and geometric constraints. The interface has two tabs: the Simulator with controls and 3D visualization, and Detailed Output showing complete analysis results.

## Controls

### Polyhedron Selection
Choose from 18 different polyhedra sorted by complexity, from simple shapes like Tetrahedron (6 edges) and Triangular Prism (9 edges) to richer forms like Pentagonal Prism (15 edges), the new family of antiprisms (16–24 edges), and the larger Dodecahedron/Icosahedron (30 edges each). The most advanced available options are Stellated Cube and Stellated Octahedron at 36 edges. Each option displays vertex count (V) and edge count (E) for reference.

### View Controls
Adjust the 3D visualization angle using Isometric (classic technical drawing), Axis-Aligned (straight-on coordinate view), or Top-Tilt (default slightly tilted view).

### Path Mode Selection
Choose your optimization strategy:

**Fixed L** finds minimum endpoints for paths of exactly length L. Use this when you need uniform path lengths. The path length input remains enabled so you can specify the desired length.

**Variable L** maximizes coverage while minimizing endpoints by trying different uniform lengths (L=1,2,3...) and picking the best. The path length input is disabled since the optimal length is automatically determined. This mode is ideal when you want optimal uniform paths but flexible about the specific length.

**Mixed lengths** provides maximum flexibility by allowing paths of different lengths within the same solution. Both path length input and fixed endpoints are disabled. This mode combines paths of various lengths (like length-1, length-2, and length-3) for ultimate optimization without constraints.

### Constraint Options

**DC only** ensures vertices are either pure Anode or pure Cathode with no alternating behavior. This enables sneak path analysis and is mutually exclusive with Alternating only mode.

**Alternating only** requires all vertices to be alternating (both Anode and Cathode), making it useful for AC-driven systems. This is mutually exclusive with DC only mode.

**Sneak path free** becomes available when DC only is selected. It ensures DC bias doesn't create shorter paths than the solver solution, which is critical for preventing current leakage.

**Equal current** forces all edges to carry exactly the same current, important for uniform LED brightness but may reduce available solutions.

**Bipolar driving scheme** ensures the solution can be implemented without tristate (Z) disconnections, using only Anode (A) and Cathode (C) assignments for simpler driver electronics.

### Advanced Settings

**Sampling iterations** controls how many random orientations to try for complex polyhedra (>12 edges). The default 2000 provides good balance of speed versus quality. Increase this for better solutions on complex shapes. Small polyhedra automatically use exhaustive search.

**Fixed endpoints** (only available in Fixed L mode) can be set to 0 to minimize endpoints (default) or to N>0 to use exactly N endpoints while maximizing coverage. This is useful when hardware has predetermined connection points.

## Understanding Results

### Solution Summary
The status line displays key metrics like "Solution found: endpoints=4, paths=6, coverage=0.833 (10/12)" where endpoints shows distinct connection points needed, paths indicates the number of LED paths in the solution, and coverage represents the fraction of edges illuminated (current/total).

### Detailed Analysis
The Detailed Output tab provides comprehensive analysis including path details, branching analysis, polarity analysis, sneak path analysis, and driving scheme analysis.

**Path Details** show the exact vertex sequence for each path like "Path 1: vertices [0, 1, 2] | len=2 | edges=[(0,1), (1,2)]" along with path length, constituent edges, and total edge coverage statistics.

**Branching Analysis** displays current flow information such as "Edge (0,1): current=0.500, branches=2, paths=[0,1] [2,3]" showing current flow through each edge, number of parallel paths (branches), and which endpoint pairs use each edge.

**Polarity Analysis** identifies the electrical role of each vertex ("Vertex 0: Anode", "Vertex 2: Cathode", "Vertex 1: Alternating") and provides design classification as either Fully DC or Mixed polarity.

**Sneak Path Analysis** (for DC only designs) validates that DC driving won't cause current leakage, showing results like "NO SNEAK PATHS: DC bias does not create shorter paths" which is critical for proper LED operation.

**Driving Scheme Analysis** specifies required Anode (A), Cathode (C), and Tristate (Z) assignments in the format "Path 1: 0 -> 2 | 0=A 1=Z 2=C # avoids sneak: 1->2". This identifies potential sneak paths that need blocking and determines whether bipolar or ternary driving is needed.


## Technical Notes

The solver uses exact cover algorithms for optimal solutions. Constraint checking is performed after path generation. The 3D visualization updates automatically with results. All algorithms preserve polyhedron connectivity, and current flow analysis assumes unit current injection. 
