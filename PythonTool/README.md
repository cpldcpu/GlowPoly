# PythonTool

Interactive simulator and solver for LED path planning on polyhedra. Note that the solver is not exhaustive, it uses a heuristic search algorithm to find a solution. Therefore it is not guaranteed to find the optimal solution. See [blog article](https://cpldcpu.github.io/2026/01/24/glowing-polyhedrons/) for more details.

**Contents:** [Overview](#overview) · [Requirements](#requirements) · [Usage](#usage) 

---

## Overview

This tool helps find driving strategies for polyhedra built from LED filaments. Given a shape, it determines how to orient the edges (anode/cathode) and where to connect feeding points so that all (or most) filaments light up.

The core challenge is that LED filaments are diodes—current flows in one direction only. On a complex 3D wireframe, not all edge orientations allow current to reach every filament. The solver searches for orientations and feeding point combinations that maximize coverage while minimizing the number of connection points.

<div align="center">
  <img src="screenshot.png" width="600px">
</div>

---

## Requirements

- Python 3.8+
- Tkinter (usually included with Python)

No external dependencies required.

---

## Usage

Run the simulator:

```bash
python glowing_polyhedron_sim.py
```

This opens an interactive GUI with:

- **Polyhedron selection** — Choose from various shapes (tetrahedron, cube, octahedron, prisms, antiprisms, etc.)
- **View controls** — Rotate the 3D visualization
- **Path mode** — Fixed length, variable length, or mixed lengths
- **Constraints** — DC only, alternating, sneak-path free, bipolar driving
- **Current mode** — Default exact cover, equal current, or greedy overlapping

The solver runs automatically when you change settings. Results show the number of endpoints needed, path coverage, and the actual driving scheme.

For detailed documentation of all controls and features, see [HELP.md](HELP.md).
