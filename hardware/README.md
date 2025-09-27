# Hardware

PCB design files and documentation for the GlowPoly LED driver board.

## Files

- `GlowPoly_Schematic_2025-09-27.pdf` - Circuit schematic
- `ProPrj_GlowPolyDriver_2025-09-27.epro` - EasyEDA project file
- `driver_pcb_render.png` - 3D PCB render
- `pcbtopdown.png` - PCB top view layout

## Overview

The driver board allows driving up to 12 feedpoints (vertices) on a polyhedron. It is controlled by a CH32V003. Each channel can be set to Anode, Cathode, or High-Z (with some constraints), allowing implementation of the various electrical configurations supported by the GlowPoly simulator software.

H-Bridge motor drivers are used to allow driving up to 10V (three strings in series). A LDO supplies 5V for the MCU.

