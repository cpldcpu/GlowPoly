
# Glowing 3D Objects from LED Filaments

*Wireframe polyhedra made entirely from LED filaments, using graph theory to optimize connection and driving strategies.*

It all began with [a video by Huy Vector](https://www.youtube.com/watch?v=zocqV4TZ4qI)[^1] that someone posted on cnlohr's Discord server: A brass wire cube with a battery and four white LED filaments. I was immediately fascinated by the idea of building objects out of LED filaments. 

But of course, in my mind, this quickly turned into a logic puzzle: Why only four filaments when a cube actually has twelve edges? What if we would build the entire cube out of LED filaments? How could we ensure that all edges light up properly? What about more interesting shapes? 

Turns out this is a deep rabbit hole. In this article, I summarize some of my findings, tools I built and suitable filaments objects, I identified.

## LED Filaments

The LED filaments commonly found in asian online market places typically measure 38mm in length with a forward voltage of 3V. Each filament consists of 16 blue LED dies mounted on a thin white ceramic substrate. All LED dies are connected in parallel. Metal contacts at both ends are serving as anode and cathode. The entire assembly is encapsulated in a silicone coating laced with a phosphor, which converts the blue light into a broader spectrum of colors. Blue filaments use a clear encapsulation without phosphor and allow us to see the individual LED dies inside.

<!-- ![LED filaments with different colors](filaments_vs_black.jpg)
![Closeup of LED filament structure](filament_close_up.jpg) -->


<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center;">
  <img src="filaments_vs_black.jpg" alt="LED filaments with different colors" style="max-width: 50%;">
</div>

<!-- <div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center;">
  <img src="filament_close_up.jpg" alt="Closeup of LED filament structure" style="max-width: 50%;">
</div>   -->


The filaments are diodes: Current can only flow in one direction, from anode to cathode. When a forward voltage of about 3V is applied, the filament lights up, with brightness proportional to the current flowing through it (typically 10-100 mA). When multiple filaments are connected in series, their forward voltages add up. Connecting them in parallel divides the current among them. I stored a more detailed characterization [here](../filaments/).

## Building Wireframe Shapes and Formalizing the Problem

With some care (and a lot of patience), the metal ends of the filaments can be soldered together to form complex 2D and 3D shapes. This can be used to build amazing glowing objects. The challenge is that also the electrical circuit is defined by how the filaments are connected in the object; a mesh that results in a mechanically stable and visually interesting structure may not necessarily represent a circuit that allows all filaments to light up properly. 

How to solve this? Let's start with a simple 2D object. The photo below shows a simple square made from four LED filaments. A constant current supply is attached to the top and bottom joint, allowing all filaments to light up. The left and right (blue and red) part of the circuit form two parallel paths for the current to flow from top to bottom, while the filaments on each side are in series. The total voltage drop is around 6V, and each filament receives about half the total current.

<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center;">
  <img src="square.jpg" alt="Square graph from simulator" style="max-width: 35%;">

  <img src="square.png" alt="Square graph from simulator" style="max-width: 35%;">
</div>

As shown on the right figure, the structure can be represented as a graph: Each junction corresponds to a **vertex**, and each filament corresponds to a connection between two vertices, a directed **edge**. Yellow markers indicate the feeding points where current is injected. This abstraction allows us to analyze the relationship between the geometry and its electrical properties using [graph theory](https://en.wikipedia.org/wiki/Graph_theory)[^2].

### What do we actually want to achieve?

Given a wireframe object made from LED filaments, what do we actually want to achieve? Two obvious objectives are:

1) All edges shall light up.
2) We want to minimize the number of feeding points $P$, the vertices where we connect the power supply.
3) The path length for all circuits between feeding points shall be exactly $L$ edges. 

The third condition is needed to ensure driving with a constant voltage supply. The voltage drop between feeding points will then equal to $V_{tot} = L \cdot V_f \approx 3V \cdot L$.

The square above meets the first condition with two feeding points and the path length is $L=2$. Things get quickly more complicated when we move to more complex objects. 

## The Cube

Let's explore a simple 3D object first: the cube. A cube has 8 connection points (vertices, $V=8$) and 12 filaments (edges, $E=12$). Each vertex connects three edges, making it a *3-regular graph*. There are trivial solutions for $L=1$ and $P=8$, where each vertex is connected to a power supply with alternating polarities. The images below show a solution as a 3D graph and a flattened 2D representation ([Schlegel diagram](https://en.wikipedia.org/wiki/Schlegel_diagram)). The number next to each edge indicates the normalized magnitude of current flow.

<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center;">
  <img src="cube_L1.png" alt="Cube graph from simulator" style="max-width: 30%;">
  <img src="cube_L1_flat.png" alt="Cube graph from simulator" style="max-width: 30%;">
</div>

Unfortunately, there is no solution that allows fewer feeding points while maintaining a constant current through all edges. However, there is a solution for $L=3$ and $P=2$, where the feeding points are connected to opposite vertices of the cube. The images below show this solution as a 3D graph and a flattened 2D representation. We can see that, while all edges are lit, the current distribution is uneven: Edges in the middle of the path only carry half of the current due to branching.

<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center;">
  <img src="cube_L3.png" alt="Cube graph (L=3)" style="max-width: 30%;">
  <img src="cube_L3_flat.png" alt="Cube Schlegel diagram (L=3)" style="max-width: 30%;">
</div>

Below you can see a photo of the actual cube with $L=3$ and two feeding points ($P=2$). While the current varies by a factor of two between edges, the impact on the appearance is surprisingly small and is rather exaggerated in the photograph. The reason for this is that we look at the filaments in separation vs. a constant brightness background. The eye response to lightness [is less than linear](https://en.wikipedia.org/wiki/Weber%E2%80%93Fechner_law), and furthermore, is mostly sensitive to [relative brightness differences](https://en.wikipedia.org/wiki/Contrast_(vision)) of object next to each other, so the filament to filament variation is less noticeable relative to a constant lightness background. 

<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center;">
  <img src="impossible_cube.jpg" alt="Cube (L=3)" style="max-width: 70%;">
</div>

## The Octahedron

The Octahedron is another simple polyhedron with $V=6$, $E=12$. In contrast to the cube, each vertex connects four edges, making it a 4-regular graph.

### DC Driving with 2 Feeding Points

No solution exists for a simple DC driving scheme where bias is applied to only two vertices (anode and cathode). As shown below in the first two images, only 8 out of 12 edges can be made to light up when driven this way. 

<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center;">
  <img src="octahedron_dc_L2.png" alt="Octahedron graph (L=2)" style="max-width: 25%;">
  <img src="octahedron_dc_l2_planar.png" alt="Octahedron Schlegel diagram (L=2)" style="max-width: 25%;">
  <img src="octahedron_dc_l2_4fp_planar.png" alt="Octahedron Schlegel diagram (L=2, 4 feeding points)" style="max-width: 25%;">
</div>

### Multiplexed DC Driving

However, as shown on in the rightmost image, if we allow four feeding points and drive them alternatingly, all edges can be made to light up. The table below shows which circuits are activated by applying a voltage to the vertices. Since vertices 0/1 feed four current paths in parallel, we need to feed in double the current as into 2/3, or keep them on for twice as long.

```
=== Driving Scheme ===
Path 1: 0 -> 2 -> 1 | 0=A 1=C
Path 2: 0 -> 3 -> 1 | 0=A 1=C
Path 3: 0 -> 4 -> 1 | 0=A 1=C
Path 4: 0 -> 5 -> 1 | 0=A 1=C
Path 5: 2 -> 4 -> 3 | 2=A 3=C
Path 6: 2 -> 5 -> 3 | 2=A 3=C
```

### Bipolar Driving 

Interestingly, there is another solution that allows driving all edges with only two feeding points: Bipolar driving, where we apply an alternating voltage.

<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center;">
  <img src="octahedron3d.png" alt="Octahedron graph (L=3)" style="max-width: 35%;">
  <img src="octahedron_ac_l3_planar.png" alt="Octahedron Schlegel diagram (L=3)" style="max-width: 35%;">
</div>

The images above show how this approach works. The path length is $L=3$, and the two feeding points are connected to opposite vertices of the octahedron. The table below shows the paths that are activated during each cycle of the alternating current driving scheme.

```
=== Driving Scheme ===
Path 1a: 0 -> 5 -> 3 -> 1 | 0=A 1=C
Path 1b: 0 -> 5 -> 2 -> 1 | 0=A 1=C
Path 2a: 0 -> 4 -> 2 -> 1 | 0=A 1=C
Path 2b: 0 -> 4 -> 3 -> 1 | 0=A 1=C
Path 3a: 1 -> 5 -> 2 -> 0 | 1=A 0=C
Path 3b: 1 -> 5 -> 3 -> 0 | 1=A 0=C
Path 4a: 1 -> 4 -> 2 -> 0 | 1=A 0=C
Path 4b: 1 -> 4 -> 3 -> 0 | 1=A 0=C
```

The four edges in the middle of the path (5->3,5->2,4->2,4->3) are part of both the forward and backward biased directions - the filaments leading to them act as a full-bridge rectifier. Since the current for the middle segments is branched (2x), the middle segments receive half the current. But since they are driven in both directions, the time-averaged current is the same as for the outer segments.

## Driver Board

Now that we moved beyond simple DC driving schemes, the question arises: How to implement this in hardware? I designed a driver board based on a CH32V003 MCU and multiple H-bridge ICs (motor driver), that can drive up to 12 feedpoints with configurable Anode, Cathode, or High-Z states at voltages up to 10V and beyond. [More details here](https://github.com/cpldcpu/GlowPoly/tree/master/hardware). Below you can see the driver board next to an Octahedron illuminated with bipolar driving.

<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center;">
  <img src="driverboard.jpg" alt="Octahedron graph (L=3)" style="max-width: 40%;">
  <img src="octahedron.jpg" alt="Octahedron Schlegel diagram (L=3)" style="max-width: 45%;">
</div>


## How to generalize this?

Assuming we have $E$ edges (filaments) and $V$ vertices (junctions),
there are $2^E$ possible ways to orient the edges. For each orientation, there are $3^V$ possible combinations to attach feeding points (each vertex can be anode, cathode, or unconnected). Searching through all combinations quickly becomes infeasible for larger objects. Even a simple cube with 12 edges and 8 vertices has $2^{12} = 4096$ possible edge orientations and $3^8 = 6561$ feeding point combinations with a total search space of over 26 million possibilities.




Given this constraint, we usually want to optimize for minimum feeding points, but I found that it often helps to set a target number of feeding points and search for valid configurations that meet this target. Note that this reduces the search space from $3^V$ to $\binom{V}{P} \cdot 2^{P}$, where $P$ is the number of feeding points.


## Euler circuit

Vertex degree must be even, usually 4.

$\frac{E}{L}=m$ feeding point pairs needed.

L even


## path

Conditions

- We have a polyhedral graph
- We want to check for the following property
- Do two points s,t exist so that all edges lies on the minimum length path between s,t.

https://en.wikipedia.org/wiki/Exact_cover

## Generalized problem statement

Let (G=(V,E)) be a (connected) polyhedral graph

For vertices (s,t\in V), let

* (d(s,t)) be the graph distance, and
* (\mathcal{P}_{\min}(s,t)) be the set of **all shortest (minimum-length) paths** from (s) to (t).


We say (G) has the property if there exist vertices (s,t) such that
[
E_{\min}(s,t) = E.
]
In words: **every edge of (G) lies on at least one shortest (s!-!t) path.**

Geodesic Cover (122 models) [taps <= 1]
=======================================
  Name                           V    E  Degrees  Path Len  Taps
  --------------------------------------------------------------
  cube                           8   12  3               3     2
  decagonal prism               20   30  3               6     2
  hexagonal prism               12   18  3               4     2
  octagonal prism               16   24  3               5     2
  square                         4    4  2               2     2
  star_octahedron               10   16  2, 4            4     2
  truncated cuboctahedron       48   72  3               9     2
  truncated icosidodecahedron  120  180  3              15     2
  truncated octahedron          24   36  3               6     2

Bidirectional Path (6 solutions) [taps <= 10]
=============================================
  Name                         V   E  Degrees  Path Len  Taps
  -----------------------------------------------------------
  cuboctahedron               12  24  4               6     2
  elongated-square-bipyramid  10  20  4               5     2
  octahedron                   6  12  4               3     2
  square                       4   4  2               2     2
  square-gyrobicupola         16  32  4               8     2
  star_octahedron             10  16  2, 4            4     2


Cycle Decomposition (12 solutions) [taps <= 10]
===============================================
  Name                       V   E  Degrees  Path Len  Taps
  ---------------------------------------------------------
  cuboctahedron             12  24  4               6     4
  icosidodecahedron         30  60  4              10     6
  octahedron                 6  12  4               4     4
  square                     4   4  2               4     2
  square-gyrobicupola       16  32  4               4    10
  star_octahedron           10  16  2, 4            4     4
  triangular-orthobicupola  12  24  4               6     5

## References and Comments

[^1]: Check out [Huy Vectors channel](https://www.youtube.com/@huyvector), he is building amazing electronic sculptures. I also learned that soldering ASMR is a thing now.

[^2]: Flash backs to CO342, arguably the least easy course I took, but also intellectually very rewarding.

## Outline






---

### 5. Solver Algorithms - Finding Solutions
- Geodesic cover problem formulation
- Algorithm approaches:
  - Exhaustive orientation search (for small graphs ≤12 edges)
  - Sampled random orientations (for large graphs)
  - Exact cover solver (using Dancing Links / bitmask approach)
- Constraint functions:
  - DC-only (no alternating vertices)
  - Sneak-free (no shorter unintended paths)
  - Bipolar-only (no tristate driving needed)
  - Equal current constraints
- **Batch Solver Pipeline** (`Solver/` directory):
  - `poly_geodesic_cover.py`: Main solver that processes 120+ polyhedra from polyhedra-viewer
  - `planar_cycle_decomposition.py`: Cycle analysis for Eulerian decomposition
  - `geodesic_cover_results.json`: 232KB of pre-computed solutions for all polyhedra
  - `visualize_geodesic_cover.py`: Visualization output generation
- Results: Processed all Johnson solids, Platonic/Archimedean solids, prisms, and antiprisms

#### Geodesic Cover vs Eulerian: What's the Difference?

**Geodesic Cover** is the approach this solver uses:
- Finds a set of **shortest paths (geodesics)** from sources to sinks that cover all edges
- Goal: Minimize the number of **feeding point pairs** (anode/cathode pairs)
- Works for **any graph structure** - doesn't require special properties

**Eulerian** is a classical graph theory concept:
- An **Eulerian path** visits every edge exactly once
- Only exists if the graph has exactly 0 or 2 odd-degree vertices
- Most polyhedra are **NOT Eulerian** (e.g., cube has all vertices of degree 3 = odd)

**How they relate:**
- If a polyhedron has an Eulerian decomposition, you can cover all edges with fewer paths
- Geodesic cover is a **more general approach** that works even when Eulerian solutions don't exist
- The "Alternating only" constraint forces vertices to act as both anode and cathode (related to Eulerian properties)

**Important tradeoff:** Some shapes can technically be covered with a single feeding point pair, but the current distribution is **uneven** - edges closer to the feedpoints carry more current than distant edges, causing brightness variation. Using more feeding pairs can achieve better current uniformity.

**Examples:**
| Shape | Vertices | Edges | Vertex Degrees | Eulerian? | Min Pairs (uneven) | Uniform Current |
|-------|----------|-------|---------------|-----------|-------------------|-----------------|
| Square | 4 | 4 | All degree-2 (even) | Yes ✓ | 1 pair | 1 pair |
| Cube | 8 | 12 | All degree-3 (odd) | No ✗ | 1 pair ⚠️ | Multiple pairs |
| Octahedron | 6 | 12 | All degree-4 (even) | Yes ✓ | 1 pair | 1-2 pairs |
| Elongated Octahedron | 10 | 18 | All degree-4 (even) | Yes ✓ | 1 pair | 1-3 pairs |
| Star Octahedron | 10 | 16 | Mixed (2 & 4, all even) | Yes ✓ | 1 pair | 1-2 pairs |
| Cuboctahedron | 12 | 24 | All degree-4 (even) | Yes ✓ | 2 pairs | 2-3 pairs |
| Hexagonal Prism | 12 | 18 | All degree-3 (odd) | No ✗ | 1 pair ⚠️ | Multiple pairs |
| Truncated Octahedron | 24 | 36 | All degree-3 (odd) | No ✗ | 1 pair ⚠️ | Multiple pairs |

---

### 6. Hardware Design - The Driver Board
- **Requirements:**
  - Drive up to 12 feedpoints (vertices) on a polyhedron
  - Each channel configurable as Anode, Cathode, or High-Z
  - Support voltage up to ~10V (3 filaments in series × 3.3V each)
- **Components:**
  - Microcontroller: CH32V003 (RISC-V, ~$0.10 cost)
  - Power stage: H-bridge motor drivers for bidirectional driving
  - Power supply: LDO for 5V MCU supply
- **PCB Design:**
  - EasyEDA project file included (`ProPrj_GlowPolyDriver_2025-09-27.epro`)
  - Schematic PDF available (`GlowPoly_Schematic_2025-09-27.pdf`)
  - Compact form factor, designed to fit inside/beside polyhedra
- *(Include: PCB 3D render, schematic overview, component list)*

---

### 7. Firmware
- **Framework:** ch32fun (lightweight CH32V003 development)
- **Architecture:**
  - GPIO outputs on PC0-PC3 and PC6-PC7 for driving channels
  - Configurable as push-pull outputs at 10MHz
  - Simple multiplexing loop with 200µs timing per step
- **Build System:** Makefile-based, generates .bin/.hex/.elf
- **Current Implementation:** Basic 4-channel test pattern
- **Future:** Full pattern sequencing based on solver output

---

### 8. Physical Construction / Assembly
- Building junctions: Soldering multiple filaments at a vertex
- Structural support: 3D printed jigs or hand assembly?
- Connecting feed wires: Thin magnet wire to driver board
- **Challenges:**
  - Thermal management (filaments get warm)
  - Mechanical fragility (glass filaments are delicate)
  - Soldering technique (needs flux, quick touch)
- *(Include photos: tetrahedron, octahedron, cuboctahedron, star octahedron, hexagonal prism, truncated octahedron)*

---

### 9. Software Tools Overview
- **PythonTool** - Interactive desktop simulator
  - `glowing_polyhedron_sim.py`: Tkinter GUI with real-time 3D visualization
  - `poly_solver.py`: Constraint-based optimization solver
  - `polyhedra.py`: Library of 20+ polyhedron generators (from tetrahedron to stellated octahedron)
  - Multiple view modes: Isometric, Graph (force-directed 2D), Schlegel diagram
  - Real-time path animation and current flow visualization
- **Solver Pipeline** - Batch processing for all polyhedra
  - Downloads models from tesseralis/polyhedra-viewer
  - Computes optimal solutions for each shape
  - Outputs JSON with solution metadata

---

### 10. Web Viewer - Interactive Visualization
- **Technology:** Three.js for 3D rendering, vanilla JavaScript
- **Live Demo:** https://cpldcpu.github.io/GlowPoly/
- **Features:**
  - Browse 120+ convex polyhedra with pre-computed solutions
  - Filter by number of feeding point pairs: [1] [2] [3] [4+] [None]
  - Color-coded path visualization with flow animation
  - Statistics panel: vertex/edge counts, coverage, solution quality
  - Flow analysis side panel with per-edge current data
- **Deployment:** GitHub Actions auto-deploys to GitHub Pages
- **Note:** Webapp mostly coded using Claude Opus 4.5!

---

### 11. Results Gallery
- Photos of completed builds:
  - Square (the simplest case - 4 edges)
  - Cube (and the "impossible cube" photo)
  - Octahedron / Elongated octahedron
  - Hexagonal prism (multiple photos)
  - Cuboctahedron (dark/light ambient photos)
  - Star octahedron
  - Truncated octahedron
- Video demonstrations showing multiplexing in action

---

### 12. Lessons Learned / Future Work
- **What worked well:**
  - Graph theory approach made the problem tractable
  - CH32V003 is cheap and capable enough
  - Web viewer makes results accessible to anyone
- **What was challenging:**
  - Soldering delicate filaments at junctions
  - Sneak path issues on complex shapes
  - Brightness uniformity with parallel paths
- **Future improvements:**
  - PWM dimming for brightness control
  - Multiplexing for complex polyhedra (>12 feedpoints)
  - Alternative LED filament types (RGB, different sizes)
  - 3D printed junction connectors
  - Smartphone app control via BLE

---

### 13. Conclusion
- Summary: From a cool video to a complete optimization + hardware system
- Open source: GitHub repo with all code, hardware files, and solver results
- Live demo: Interactive web viewer anyone can use
- Call to action: Build your own glowing polyhedron!



# Driving Strategies: Bipolar vs Eulerian

Not all solutions are created equal. Depending on your hardware constraints, you may prefer different approaches to "driving" the polyhedron - that is, how you assign anode (+) and cathode (-) to the vertices.

## Bipolar Driving

The simplest approach: use exactly **2 feeding points** (one anode, one cathode). This is called "bipolar" because you only need two polarities - no need for high-impedance (Hi-Z) states.

**Advantages:**
- Simple driver hardware (just needs A and C, no tristate switching)
- Easy to understand and implement

**Disadvantages:**
- Not all polyhedra can be covered with just 2 feeding points
- May require longer path lengths (higher voltage)

**Example:** The square is naturally bipolar. With v0 as anode and v3 as cathode, current flows through all 4 edges via two parallel paths.

## Eulerian / Alternating Driving

A more sophisticated approach: allow vertices to act as **both anode AND cathode** at different times. This is related to Eulerian paths in graph theory - you're essentially traversing every edge exactly once.

**Advantages:**
- Perfect for AC driving (reversing polarity periodically)
- Can achieve more uniform current distribution
- Often covers all edges with fewer path problems

**Disadvantages:**
- Requires more complex driving hardware (must switch polarity)
- Needs careful timing/sequencing



## Which to Choose?

| Approach | Feeding Points | Hardware | Best For |
|----------|---------------|----------|----------|
| Bipolar | 2 | Simple (A/C only) | Small, simple shapes |
| Multi-pair | 4+ | Complex (A/C/Z) | Larger polyhedra |
| Alternating | Varies | AC driver | Uniform brightness, complex shapes |

---

# LED Filaments

LED filaments are the building blocks of this project. Understanding their properties is essential.

<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center; margin: 1rem 0;">
  <img src="../media/filaments1.JPG" alt="LED filaments" style="max-width: 45%;">
  <img src="../media/filaments2.JPG" alt="LED filaments closeup" style="max-width: 45%;">
</div>

- **Construction:** ~16 LED dies connected in parallel on a phosphor-coated substrate
- **Electrical:** ~3V forward voltage, light output proportional to current (up to ~150mA)
- **Behavior:** Current flows only in one direction (diode), voltages add in series, current divides in parallel

For detailed measurement data and I-V characteristics, see the [filaments](../filaments/) folder.

---

# Building Polyhedra


## The Cube

The cube is a 3-regular graph with 8 vertices and 12 edges. Each junction needs to accommodate three filaments.

<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center;">
  <img src="cube3d.png" alt="Cube 3D view" style="max-width: 30%;">
  <img src="cube2d.png" alt="Cube 2D graph" style="max-width: 30%;">
  <img src="../media/impossible_cube.jpg" alt="Impossible cube" style="max-width: 30%;">
</div>

## The Octahedron

Real builds - note the elongated octahedron is a different shape (regular octahedron with a square prism inserted):

<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center;">
  <figure style="text-align: center; max-width: 55%;">
    <img src="../media/octahedron.jpg" alt="Octahedron build" style="width: 100%;">
    <figcaption>Regular Octahedron (6 vertices, all degree-4)</figcaption>
  </figure>
  <figure style="text-align: center; max-width: 40%;">
    <img src="../media/elongated%20octahedron.jpg" alt="Elongated octahedron" style="width: 100%;">
    <figcaption>Elongated Octahedron (10 vertices, all degree-4)</figcaption>
  </figure>
</div>

---

# Hardware

## The Driver Board

The GlowPoly driver board controls up to 12 feedpoints using H-bridge motor drivers and a CH32V003 microcontroller.

<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center;">
  <img src="../media/driverboard.jpg" alt="GlowPoly driver board" style="max-width: 50%;">
</div>

---

# Results Gallery

## Hexagonal Prism

<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center; margin: 1rem 0;">
  <img src="../media/hexagonal%20prism%202.JPG" alt="Hexagonal prism" style="max-width: 30%;">
  <img src="../media/hexagonal%20prism%203.JPG" alt="Hexagonal prism angle 2" style="max-width: 30%;">
  <img src="../media/hexagonal%20prism%20hand.JPG" alt="Hexagonal prism in hand" style="max-width: 30%;">
</div>

Uneven brightness demonstration (showing parallel path current distribution):

![Hexagonal prism uneven brightness](../media/hexagonal%20prism%20uneven.JPG)

## Cuboctahedron

The cuboctahedron (12 vertices, 24 edges) is one of the more complex builds.

<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center; margin: 1rem 0;">
  <img src="../media/cuboctahedron_dark.jpg" alt="Cuboctahedron dark background" style="max-width: 45%;">
  <img src="../media/cuboctahedron_light.jpg" alt="Cuboctahedron light background" style="max-width: 45%;">
</div>

Hero shot:

![Cuboctahedron 16:9](../media/cuboctahedron_16_9.jpg)

## Star Octahedron

<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center; margin: 1rem 0;">
  <img src="../media/star%20octahedron%202.JPG" alt="Star octahedron" style="max-width: 45%;">
  <img src="../media/star%20hand.JPG" alt="Star in hand" style="max-width: 45%;">
</div>

## Truncated Octahedron

![Truncated octahedron](../media/truncated%20octahedron.JPG)

