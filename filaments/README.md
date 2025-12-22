# LED Filaments

<div align="center">
  <img src="filament_close_up.jpg" height="200px">
  <img src="filament_anode.jpg" height="200px">
  <img src="filament_lit.jpg" height="200px">
</div>

These are the COB LED strips found in vintage-style Edison bulbs. About 16 LED dies sit in parallel on a phosphor-coated glass substrate, ~38mm long. The phosphor converts blue to warm white.

## Electrical Characterization

I connected a filament to a bench supply and measured voltage and light output (using an ambient light sensor) across the operating range.

![I-V Curve](filament_iv_curve.png)

| I [mA] | V | Radiance (vs 100mA) |
|--------|------|---------------------|
| 10 | 2.59 | 0.11 |
| 50 | 2.75 | 0.52 |
| 100 | 2.91 | 1.00 |
| 150 | 3.06 | 1.44 |

Forward voltage goes from 2.6V at 10mA to 3.1V at 150mA. Radiance scales linearly with current - no surprises there.

## Implications for Polyhedra

Series: Voltages add. Three filaments need ~9V, which is close to the 10V limit of the H-bridge drivers.

Parallel: Current divides. This is where the graph theory comes in - current distribution through a mesh of parallel paths is the core problem the solver addresses.

The linear radiance/current relationship means brightness uniformity depends entirely on current uniformity. Achieving that in complex polyhedra is non-trivial.
