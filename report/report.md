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

Given a wireframe object made from LED filaments, what do we actually want to achieve? Some straightforward objectives:

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

### Multiplexed DC Driving with 4 feeding points

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

### Bipolar Driving with 2 feeding points

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

## Which other filament objects can we build?

The examples above were found through intuition and manual exploration. But what about more complex objects? Which are eligible and how to orient edges so that we can optimize the driving scheme?

A [polyhedron](https://en.wikipedia.org/wiki/Polyhedron) (plural: polyhedra) is a three-dimensional solid with flat polygonal faces and straight edges. The simplest examples are the five Platonic solids: tetrahedron, cube, octahedron, dodecahedron, and icosahedron. Beyond these there are hundreds polyhedra that have been catalogues by mathematician. The [Polyhedra Viewer by Nat Alison](https://polyhedra.tessera.li/) provides a nice interactive catalog, and is also a good source of [3D object files](https://github.com/tesseralis/polyhedra-viewer/tree/canon/src/data/polyhedra). 

 For our purposes, each polyhedron defines a filament object: vertices become electrical junctions, edges become LED filaments. 
 
### Finding solutions for any shape - randomized search

Initially, I "vibe coded" a visualizer and solver in python to find solutions for a given object, given constraints and visualize it. You can [find it here](https://github.com/cpldcpu/GlowPoly/tree/master/PythonTool). 

However, I had to learn that a randomized search is not sufficient to find good solutions for larger objects due to an enormous explosion of the search space. Given a polyhedron with $E$ edges and $V$ vertices, there are $2^E$ possible ways to orient the edges (assigning anode/cathode direction to each filament). For each orientation, there are $3^V$ possible feeding point combinations (each vertex can be anode, cathode, or floating). This quickly becomes infeasible, even a simple cube has $2^{12} \cdot 3^8 \approx 26$ million possibilities. 

Instead, it is necessary to reduce the search space by limiting the number of options looked at. I found the most feasible approach was to assume certain driving schemes, as identified in the manual search above, and try to identify eligible objects for it.

Based on the exploration of the Octahedron and Cube above, we can observe three different configurations of filaments that allow driving an entire polyhedron from a few vertices:

1. DC from two feeding points, ignoring brightness variation (as seen in the cube)
2. Multiplexing several DC paths to achieve constant brightness (as seen in the octahedron)
3. Bipolar driving of two feeding points with constant brightness (octahedron)

How do we screen for objects that support these driving schemes?

### Case 1: DC driving from two feeding points, Edge-Geodesic Cover

Just to recap: We want to connect a positive terminal (anode) to one vertex and a negative terminal (cathode) to another vertex, so that all filaments (edges) lie on the current path between these two points. For all filaments to light up, every edge must lie on a shortest path between the two feeding points. We then try to identify such pairs of vertices, so that the shortest paths between them cover all edges. Note that this will result in a solution that allows all filaments to light up, but since there can (and will) be branching paths, the current distribution will be uneven.

This (and it took me quite a while to figure this out) is known in graph theory as  **edge-geodesic cover problem**. Specifically, we want to screen polyhedral graphs for the existence of an *edge-geodetic set with an edge geodetic number of 2*. This means that there exist two vertices (s,t) such that every edge in the graph lies on some shortest path between s and t. A necessary precondition for this to be possible is that the graph is [bipartite](https://en.wikipedia.org/wiki/Bipartite_graph), which can be easily tested for. Essentially that states that the graph can be separated layer-by-layer without any dangling edges (filaments) between the layers.

Only 6 out of the 122 polyhedra I analyzed support this property and allow driving all filaments from only two connections. They are listed in the table below.

| Name                         | $V$   | $E$   | $deg(V)$ | $L$ 
|------------------------------|-----|-----|---------|----------
| cube                         | 8   | 12  | 3       | 3        
| hexagonal prism              | 12  | 18  | 3       | 4        
| octagonal prism              | 16  | 24  | 3       | 5        
| decagonal prism              | 20  | 30  | 3       | 6        
| truncated octahedron         | 24  | 36  | 3       | 6        
| truncated cuboctahedron      | 48  | 72  | 3       | 9        
| truncated icosidodecahedron  | 120 | 180 | 3       | 15       

I built two of those objects: The hexagonal prism and the truncated octahedron. Below you can see photos of both objects in all their glory.

<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center;">
  <img src="hexagonal prism 3.JPG" alt="Hexagonal prism" style="max-width: 30%;">
  <img src="hexagonal prism hand.JPG" alt="Hexagonal prism" style="max-width: 30%;">
  <img src="truncated octahedron.JPG" alt="Truncated octahedron" style="max-width: 30%;">
</div>

However, all objects allow solutions where the filaments can be covered by shortest-paths between multiple combinations of feeding point pairs. This allows for DC solutions with multiplexing. I found that many objects require 4 taps, while the most taps needed at 8. You can explore further geodesic cover solutions in the [interactive web viewer](https://cpldcpu.github.io/GlowPoly/).

### Case 2: Eulerian Circuits and Cycle Decomposition

While the previous approach allows finding solutions that can light up all filaments, the current distribution is usually uneven.



Mind you, while the edge-geodesic cover solutions above allow driving all filaments from two feeding points, the current distribution is uneven. To achieve more uniform brightness, we can explore other driving schemes.

The second driving scheme is based on the idea of allowing vertices to act as both anode and cathode at different times. This is related to Eulerian paths in graph theory - you're essentially traversing every edge exactly once.

https://en.wikipedia.org/wiki/Euler_characteristic


## Euler circuit

Vertex degree must be even, usually 4.

$\frac{E}{L}=m$ feeding point pairs needed.

L even

**Eulerian** is a classical graph theory concept:
- An **Eulerian path** visits every edge exactly once
- Only exists if the graph has exactly 0 or 2 odd-degree vertices
- Most polyhedra are **NOT Eulerian** (e.g., cube has all vertices of degree 3 = odd)


Bidirectional Path (6 solutions) [taps <= 10]
=============================================
  Name                         V   E  Degrees  Path Len  Taps
  -----------------------------------------------------------
  cuboctahedron               12  24  4               6     2
  elongated-square-bipyramid  10  20  4               5     2
  octahedron                   6  12  4               3     2
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


## Cuboctahedron

The cuboctahedron (12 vertices, 24 edges) is one of the more complex builds.

<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center; margin: 1rem 0;">
  <img src="../media/cuboctahedron_dark.jpg" alt="Cuboctahedron dark background" style="max-width: 45%;">
  <img src="../media/cuboctahedron_light.jpg" alt="Cuboctahedron light background" style="max-width: 45%;">
</div>


![Cuboctahedron 16:9](../media/cuboctahedron_16_9.jpg)

## Star Octahedron

<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center; margin: 1rem 0;">
  <img src="../media/star%20octahedron%202.JPG" alt="Star octahedron" style="max-width: 45%;">
  <img src="../media/star%20hand.JPG" alt="Star in hand" style="max-width: 45%;">
</div>

