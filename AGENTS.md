# Repository Guidelines

## Project Structure & Module Organization
Source scripts live at the repository root: `partial_coverage2.py` explores k-terminal geodesic coverage and prints detailed reports for several polyhedra, while `poly_solver.py` experiments with fixed-length objective variants. Shared undirected graph builders sit in `polyhedra.py` so both entry points consume the same generators. Desktop tools `display_polyhedra.py`, `glowing_polyhedron_sim.py`, and `polylight_gui.py` provide interactive views; the last two wrap optimization solvers in GUI front ends. No generated assets are checked in—command output streams to STDOUT, so capture runs with shell redirection when you need an audit trail.

## Build, Test, and Development Commands
Use Python 3.10+ and install optional tooling in a virtual environment: `python -m venv .venv && source .venv/bin/activate`. Run the primary demo suite with `python partial_coverage2.py`; exercise the alternate objective with `python poly_solver.py`. For quick profiling, wrap executions with `python -m cProfile partial_coverage2.py > profile.txt` to keep performance notes alongside results.

## Coding Style & Naming Conventions
Follow PEP 8 defaults: 4-space indentation, snake_case for functions, PascalCase reserved for classes such as `DiGraph`. Keep module-level constants uppercase. Type hints are encouraged on new code paths, especially around graph primitives. Retain the existing docstring pattern that explains objectives and solver strategy near the top of each module, and prefer succinct inline comments only when an algorithmic decision is non-obvious.

## Testing Guidelines
Add regression coverage with `pytest`; place suites under `tests/` mirroring the module names (for example `tests/test_partial_coverage.py`). `tests/test_polyhedra.py` exercises the shared generators; extend it when new shapes are added. Focus on verifying graph builders, geodesic enumeration, and optimization edge cases (e.g., disconnected orientations, large k). Include minimal fixtures that construct synthetic graphs, and assert both coverage counts and uncovered-edge listings. Run `pytest -q` before submitting changes.

## Commit & Pull Request Guidelines
The repository has no formal history yet; start commits with an imperative summary (`solve: optimize k-terminal search pruning`) and treat the body as a concise changelog. Reference related issues or datasets in bullet form. Pull requests should restate the objective, list verification commands, and attach sample output snippets when solver behavior changes so reviewers can diff coverage improvements easily.
