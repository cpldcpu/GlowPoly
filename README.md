# GlowPoly

Building wireframe polyhedra from LED filaments, using graph theory to find driving strategies.

<div align="center">
  <img src="media/featured.jpg" width="700px">
</div>

## About

This project explores building glowing 3D wireframe objects from LED filaments. The challenge is that filaments are diodes, current flows one way only, and the electrical circuit is defined by how you solder them together. Not many shapes can be fully illuminated from an acceptable number of feeding points. 


**[Read the full blog post](https://cpldcpu.github.io/2026/01/24/glowing-polyhedrons/)**

**[Try the interactive web viewer](https://cpldcpu.github.io/GlowPoly/)**

<div align="center">
  <img src="media/webviewer.png" width="700px">
</div>

---

## Repository Structure

### [hardware/](hardware/)
PCB design for the GlowPoly driver board. Uses a CH32V003 MCU and H-bridge motor drivers to control up to 12 feeding points with configurable anode/cathode/high-Z states at up to 10V or more depending on the H-bridge drivers used.

### [hardware/filaments/](hardware/filaments/)
Documentation on the LED filaments: electrical characterization, I-V curves, and soldering tips for building polyhedra.

### [PythonTool/](PythonTool/)
Interactive simulator with a Tkinter GUI. Visualizes polyhedra, runs solvers, and shows driving schemes. Good for exploring individual shapes.

### [Solver/](Solver/)
Batch solvers for analyzing many polyhedra at once:
- `poly_geodesic_cover.py`  Edge-geodesic cover solver (DC driving from 2+ points)
- `equal_cycle_decomp.py`  Cycle decomposition for uniform current distribution
- `bidirectional_path_decomp.py`  Bipolar/AC driving analysis
- `model_stats.py`  Statistics across all polyhedra models

### [webviewer/](webviewer/)
Three.js-based web viewer for exploring solutions interactively. Shows 3D models with current flow visualization, path highlighting, and VR support. Deployed at [cpldcpu.github.io/GlowPoly](https://cpldcpu.github.io/GlowPoly/).

### [firmware/](firmware/)
CH32V003 firmware for the driver board. Cycles through driving schemes to multiplex current paths. Built on [ch32fun](https://github.com/cnlohr/ch32fun).

### [media/](media/)
Photos of built polyhedra: cubes, prisms, octahedra, star shapes, and more.
