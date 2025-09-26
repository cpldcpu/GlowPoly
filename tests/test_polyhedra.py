import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyhedra import PolyhedronGenerators

GENERATOR_EXPECTATIONS = [
    ("undirected_tetrahedron", 4, 6),
    ("undirected_cube", 8, 12),
    ("undirected_octahedron", 6, 12),
    ("undirected_cuboctahedron", 12, 24),
    ("undirected_rhombic_dodecahedron", 14, 24),
    ("undirected_truncated_tetrahedron", 12, 18),
    ("undirected_stellated_tetrahedron", 8, 18),
    ("undirected_dodecahedron", 20, 30),
    ("undirected_icosahedron", 12, 30),
    ("undirected_stellated_cube", 14, 36),
    ("undirected_stellated_octahedron", 14, 36),
    ("undirected_stellated_triangular_prism", 11, 27),
]


@pytest.mark.parametrize("name, n_vertices, n_edges", GENERATOR_EXPECTATIONS)
def test_polyhedron_generators(name, n_vertices, n_edges):
    builder = getattr(PolyhedronGenerators, name)
    vertices, edges = builder()

    assert len(vertices) == n_vertices
    assert set(vertices) == set(range(n_vertices))

    assert len(edges) == n_edges
    assert len(set(edges)) == len(edges)

    for u, v in edges:
        assert isinstance(u, int) and isinstance(v, int)
        assert 0 <= u < n_vertices and 0 <= v < n_vertices
        assert u < v  # undirected edges stored in ascending order
