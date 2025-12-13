"""Shared polyhedron graph generators for geodesic coverage solvers."""

from __future__ import annotations

from math import sqrt


class PolyhedronGenerators:
    """Build undirected polyhedron graphs as (vertices, edges) tuples."""

    @staticmethod
    def _finalize(vertices, edges, coords, return_coords):
        edges_sorted = sorted(edges)
        return (vertices, edges_sorted, coords) if return_coords else (vertices, edges_sorted)

    @staticmethod
    def undirected_tetrahedron(return_coords: bool = False):
        coords = [
            (1, 1, 1),
            (1, -1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
        ]
        vertices = list(range(4))
        edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_cube(return_coords: bool = False):
        vertices = list(range(8))
        coords = [
            (
                1 if (idx & 1) else -1,
                1 if (idx & 2) else -1,
                1 if (idx & 4) else -1,
            )
            for idx in vertices
        ]
        edges = set()
        for u in vertices:
            for bit in (1, 2, 4):
                v = u ^ bit
                if u < v:
                    edges.add((u, v))
                else:
                    edges.add((v, u))
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_octahedron(return_coords: bool = False):
        coords = [
            (0, 0, 1),
            (0, 0, -1),
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
        ]
        vertices = list(range(6))
        forbidden = {(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)}
        edges = set()
        for u in vertices:
            for v in vertices:
                if u < v and (u, v) not in forbidden:
                    edges.add((u, v))
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_cuboctahedron(return_coords: bool = False):
        coords = []
        for a in (-1, 1):
            for b in (-1, 1):
                coords.append((a, b, 0))
                coords.append((a, 0, b))
                coords.append((0, a, b))
        vertices = list(range(12))
        edges = set()
        for i in range(12):
            for j in range(i + 1, 12):
                if PolyhedronGenerators._dist2(coords[i], coords[j]) == 2:
                    edges.add((i, j))
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_rhombic_dodecahedron(return_coords: bool = False):
        coords = []
        for x in (-1, 1):
            for y in (-1, 1):
                for z in (-1, 1):
                    coords.append((x, y, z))
        coords.extend([(2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2)])
        vertices = list(range(14))
        edges = set()
        target_dist2 = 3  # cube vertices connect to adjacent face centers
        for i in range(14):
            for j in range(i + 1, 14):
                if abs(PolyhedronGenerators._dist2(coords[i], coords[j]) - target_dist2) < 1e-10:
                    edges.add((i, j))
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_truncated_tetrahedron(return_coords: bool = False):
        a = 3.0
        b = 1.0
        coords = [
            (a, b, b), (b, a, b), (b, b, a),
            (a, -b, -b), (b, -a, -b), (b, -b, -a),
            (-a, b, -b), (-b, a, -b), (-b, b, -a),
            (-a, -b, b), (-b, -a, b), (-b, -b, a),
        ]
        vertices = list(range(12))
        edges = set()
        target_dist2 = 8.0
        for i in range(12):
            for j in range(i + 1, 12):
                if abs(PolyhedronGenerators._dist2(coords[i], coords[j]) - target_dist2) < 1e-10:
                    edges.add((i, j))
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_dodecahedron(return_coords: bool = False):
        phi = (1 + sqrt(5)) / 2
        coords = []
        for x in (-1, 1):
            for y in (-1, 1):
                for z in (-1, 1):
                    coords.append((x, y, z))
        for x in (-1 / phi, 1 / phi):
            for y in (-phi, phi):
                coords.append((x, y, 0))
                coords.append((y, 0, x))
                coords.append((0, x, y))
        vertices = list(range(20))
        edges = set()
        target_dist2 = (2 / phi) ** 2
        for i in range(20):
            for j in range(i + 1, 20):
                if abs(PolyhedronGenerators._dist2(coords[i], coords[j]) - target_dist2) < 0.1:
                    edges.add((i, j))
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_icosahedron(return_coords: bool = False):
        phi = (1 + sqrt(5)) / 2
        coords = []
        for x in (-1, 1):
            for y in (-phi, phi):
                coords.append((x, y, 0))
                coords.append((0, x, y))
                coords.append((y, 0, x))
        vertices = list(range(12))
        edges = set()
        target_dist2 = 4.0
        for i in range(12):
            for j in range(i + 1, 12):
                if abs(PolyhedronGenerators._dist2(coords[i], coords[j]) - target_dist2) < 0.1:
                    edges.add((i, j))
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_triangular_prism(return_coords: bool = False):
        # Top triangle at z=1, bottom triangle at z=-1
        coords = [
            # Top triangle (z=1)
            (1, 0, 1),
            (-0.5, sqrt(3)/2, 1),
            (-0.5, -sqrt(3)/2, 1),
            # Bottom triangle (z=-1)
            (1, 0, -1),
            (-0.5, sqrt(3)/2, -1),
            (-0.5, -sqrt(3)/2, -1),
        ]
        vertices = list(range(6))
        edges = set()
        # Connect triangular faces
        edges.update([(0, 1), (1, 2), (2, 0)])  # top triangle
        edges.update([(3, 4), (4, 5), (5, 3)])  # bottom triangle
        # Connect corresponding vertices between top and bottom
        edges.update([(0, 3), (1, 4), (2, 5)])
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_pentagonal_prism(return_coords: bool = False):
        from math import cos, sin, pi
        # Top pentagon at z=1, bottom pentagon at z=-1
        coords = []
        for i in range(5):
            angle = 2 * pi * i / 5
            x, y = cos(angle), sin(angle)
            coords.append((x, y, 1))  # top pentagon
        for i in range(5):
            angle = 2 * pi * i / 5
            x, y = cos(angle), sin(angle)
            coords.append((x, y, -1))  # bottom pentagon
        
        vertices = list(range(10))
        edges = set()
        # Connect pentagonal faces
        for i in range(5):
            edges.add((i, (i + 1) % 5))  # top pentagon
            edges.add((i + 5, ((i + 1) % 5) + 5))  # bottom pentagon
            edges.add((i, i + 5))  # vertical edges connecting top to bottom
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_hexagonal_prism(return_coords: bool = False):
        from math import cos, sin, pi
        # Top hexagon at z=1, bottom hexagon at z=-1
        coords = []
        for i in range(6):
            angle = 2 * pi * i / 6
            x, y = cos(angle), sin(angle)
            coords.append((x, y, 1))  # top hexagon
        for i in range(6):
            angle = 2 * pi * i / 6
            x, y = cos(angle), sin(angle)
            coords.append((x, y, -1))  # bottom hexagon
        
        vertices = list(range(12))
        edges = set()
        # Connect hexagonal faces
        for i in range(6):
            edges.add((i, (i + 1) % 6))  # top hexagon
            edges.add((i + 6, ((i + 1) % 6) + 6))  # bottom hexagon
            edges.add((i, i + 6))  # vertical edges connecting top to bottom
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def _undirected_antiprism(sides: int, return_coords: bool = False):
        """Build an n-gonal antiprism made of two offset polygons."""
        if sides < 3:
            raise ValueError("Antiprism requires at least 3 sides")

        from math import cos, sin, pi

        coords = []
        top_z, bottom_z = 1.0, -1.0
        for i in range(sides):
            angle = 2 * pi * i / sides
            coords.append((cos(angle), sin(angle), top_z))
        for i in range(sides):
            angle = 2 * pi * i / sides + pi / sides  # rotate bottom by half-step
            coords.append((cos(angle), sin(angle), bottom_z))

        vertices = list(range(2 * sides))
        edges = set()
        for i in range(sides):
            edges.add((i, (i + 1) % sides))  # top ring
            edges.add((sides + i, sides + ((i + 1) % sides)))  # bottom ring
            edges.add((i, sides + i))  # lateral triangle edge
            edges.add((i, sides + ((i - 1) % sides)))  # lateral triangle edge
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_square_antiprism(return_coords: bool = False):
        return PolyhedronGenerators._undirected_antiprism(4, return_coords)

    @staticmethod
    def undirected_pentagonal_antiprism(return_coords: bool = False):
        return PolyhedronGenerators._undirected_antiprism(5, return_coords)

    @staticmethod
    def undirected_hexagonal_antiprism(return_coords: bool = False):
        return PolyhedronGenerators._undirected_antiprism(6, return_coords)

    @staticmethod
    def undirected_triangular_orthobicupola(return_coords: bool = False):
        """Triangular orthobicupola (J27): 12 vertices, 24 edges.

        Unlike the cuboctahedron (triangular gyrobicupola) where the two triangles
        are rotated 60° from each other, in the orthobicupola the triangles are
        in the SAME orientation (aligned).
        """
        from math import cos, sin, pi, sqrt

        # Two triangular cupolas joined at hexagonal bases WITHOUT rotation
        r = 1 / sqrt(3)  # Triangle circumradius for unit edge
        h = sqrt(2 / 3)  # Height of triangle above/below hexagon

        coords = []

        # Top triangle (z = h), at angles 30°, 150°, 270°
        for k in range(3):
            angle = pi / 6 + 2 * pi * k / 3
            coords.append((r * cos(angle), r * sin(angle), h))

        # Middle hexagon (z = 0), at angles 0°, 60°, 120°, 180°, 240°, 300°
        for k in range(6):
            angle = pi * k / 3
            coords.append((cos(angle), sin(angle), 0))

        # Bottom triangle (z = -h), SAME orientation as top (ortho = aligned)
        for k in range(3):
            angle = pi / 6 + 2 * pi * k / 3  # Same angles as top triangle
            coords.append((r * cos(angle), r * sin(angle), -h))

        vertices = list(range(12))
        edges = set()

        # Use distance-based edge detection (edge length = 1)
        target_dist2 = 1.0
        for i in range(12):
            for j in range(i + 1, 12):
                dist2 = PolyhedronGenerators._dist2(coords[i], coords[j])
                if abs(dist2 - target_dist2) < 0.01:
                    edges.add((i, j))

        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_elongated_octahedron(return_coords: bool = False):
        """Elongated square bipyramid (J15): 10 vertices, 20 edges.

        A square prism with square pyramids attached to both ends.
        """
        from math import sqrt

        coords = []

        # Bottom apex
        coords.append((0, 0, -2))

        # Bottom square (z = -1)
        coords.append((1, 0, -1))
        coords.append((0, 1, -1))
        coords.append((-1, 0, -1))
        coords.append((0, -1, -1))

        # Top square (z = 1)
        coords.append((1, 0, 1))
        coords.append((0, 1, 1))
        coords.append((-1, 0, 1))
        coords.append((0, -1, 1))

        # Top apex
        coords.append((0, 0, 2))

        vertices = list(range(10))
        edges = set()

        # Bottom apex (0) to bottom square (1-4)
        for i in range(1, 5):
            edges.add((0, i))

        # Bottom square edges
        edges.add((1, 2))
        edges.add((2, 3))
        edges.add((3, 4))
        edges.add((4, 1))

        # Vertical edges connecting squares
        edges.add((1, 5))
        edges.add((2, 6))
        edges.add((3, 7))
        edges.add((4, 8))

        # Top square edges
        edges.add((5, 6))
        edges.add((6, 7))
        edges.add((7, 8))
        edges.add((8, 5))

        # Top apex (9) to top square (5-8)
        for i in range(5, 9):
            edges.add((i, 9))

        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def _undirected_bipyramid(sides: int, return_coords: bool = False):
        """Build an n-gonal bipyramid (two pyramids joined at base)."""
        if sides < 3:
            raise ValueError("Bipyramid requires at least 3 sides")

        from math import cos, sin, pi

        coords = []

        # Bottom apex
        coords.append((0, 0, -1))

        # Middle polygon (z = 0)
        for i in range(sides):
            angle = 2 * pi * i / sides
            coords.append((cos(angle), sin(angle), 0))

        # Top apex
        coords.append((0, 0, 1))

        vertices = list(range(sides + 2))
        edges = set()

        # Bottom apex (0) to polygon vertices (1 to sides)
        for i in range(1, sides + 1):
            edges.add((0, i))

        # Polygon edges
        for i in range(1, sides + 1):
            next_i = (i % sides) + 1
            edges.add((i, next_i))

        # Top apex (sides+1) to polygon vertices (1 to sides)
        top_apex = sides + 1
        for i in range(1, sides + 1):
            edges.add((i, top_apex))

        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_hexagonal_bipyramid(return_coords: bool = False):
        """Hexagonal bipyramid: 8 vertices, 18 edges."""
        return PolyhedronGenerators._undirected_bipyramid(6, return_coords)

    @staticmethod
    def undirected_octagonal_bipyramid(return_coords: bool = False):
        """Octagonal bipyramid: 10 vertices, 24 edges."""
        return PolyhedronGenerators._undirected_bipyramid(8, return_coords)

    @staticmethod
    def undirected_elongated_hexagonal_bipyramid(return_coords: bool = False):
        """Elongated hexagonal bipyramid: 14 vertices, 30 edges.

        A hexagonal prism with hexagonal pyramids attached to both ends.
        """
        from math import cos, sin, pi

        coords = []

        # Bottom apex
        coords.append((0, 0, -2))

        # Bottom hexagon (z = -1)
        for i in range(6):
            angle = 2 * pi * i / 6
            coords.append((cos(angle), sin(angle), -1))

        # Top hexagon (z = 1)
        for i in range(6):
            angle = 2 * pi * i / 6
            coords.append((cos(angle), sin(angle), 1))

        # Top apex
        coords.append((0, 0, 2))

        vertices = list(range(14))
        edges = set()

        # Bottom apex (0) to bottom hexagon (1-6)
        for i in range(1, 7):
            edges.add((0, i))

        # Bottom hexagon edges (vertices 1-6)
        for i in range(6):
            edges.add((i + 1, (i + 1) % 6 + 1))
        # Explicit for clarity
        edges.add((1, 2))
        edges.add((2, 3))
        edges.add((3, 4))
        edges.add((4, 5))
        edges.add((5, 6))
        edges.add((6, 1))

        # Vertical edges connecting hexagons (1-6 to 7-12)
        for i in range(6):
            edges.add((i + 1, i + 7))

        # Top hexagon edges (vertices 7-12)
        edges.add((7, 8))
        edges.add((8, 9))
        edges.add((9, 10))
        edges.add((10, 11))
        edges.add((11, 12))
        edges.add((12, 7))

        # Top apex (13) to top hexagon (7-12)
        for i in range(7, 13):
            edges.add((i, 13))

        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_rhombicuboctahedron(return_coords: bool = False):
        # Rhombicuboctahedron coordinates: (±1, ±1, ±(1+√2)) and permutations
        s = 1 + sqrt(2)
        coords = []
        # All permutations of (±1, ±1, ±s)
        for x in (-1, 1):
            for y in (-1, 1):
                for z in (-s, s):
                    coords.append((x, y, z))
                    coords.append((x, z, y))
                    coords.append((y, x, z))
        
        vertices = list(range(24))
        edges = set()
        # Connect vertices that are at distance sqrt(2) or 2
        for i in range(24):
            for j in range(i + 1, 24):
                dist2 = PolyhedronGenerators._dist2(coords[i], coords[j])
                if abs(dist2 - 2.0) < 0.1 or abs(dist2 - 4.0) < 0.1:
                    edges.add((i, j))
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_rhombicosidodecahedron(return_coords: bool = False):
        # Rhombicosidodecahedron has 60 vertices
        phi = (1 + sqrt(5)) / 2  # golden ratio
        coords = []
        
        # Even permutations of (±1, ±1, ±φ³)
        phi3 = phi ** 3
        for x in (-1, 1):
            for y in (-1, 1):
                for z in (-phi3, phi3):
                    coords.append((x, y, z))
                    coords.append((y, z, x))
                    coords.append((z, x, y))
        
        # Even permutations of (±φ², ±φ, ±2φ)
        phi2 = phi ** 2
        phi2_2 = 2 * phi
        for x in (-phi2, phi2):
            for y in (-phi, phi):
                for z in (-phi2_2, phi2_2):
                    coords.append((x, y, z))
                    coords.append((y, z, x))
                    coords.append((z, x, y))
        
        # Even permutations of (±(2φ+1), ±φ², ±1)
        phi2_plus1 = 2 * phi + 1
        for x in (-phi2_plus1, phi2_plus1):
            for y in (-phi2, phi2):
                for z in (-1, 1):
                    coords.append((x, y, z))
                    coords.append((y, z, x))
                    coords.append((z, x, y))
        
        vertices = list(range(60))
        edges = set()
        # Connect vertices at appropriate distances (this is approximate)
        target_dist2 = 4.0  # Approximate edge length squared
        for i in range(60):
            for j in range(i + 1, 60):
                dist2 = PolyhedronGenerators._dist2(coords[i], coords[j])
                if abs(dist2 - target_dist2) < 1.0:  # Allow some tolerance
                    edges.add((i, j))
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_snub_cube(return_coords: bool = False):
        # Snub cube coordinates using tribonacci constant
        # ξ is the real root of x³ - x² - x - 1 = 0 (tribonacci constant)
        # Approximate value: ξ ≈ 1.839
        xi = 1.839286755214161  # tribonacci constant
        coords = []

        # All even permutations of (±1, ±ξ, ±1/ξ)
        xi_inv = 1 / xi
        perms = [
            (1, xi, xi_inv), (xi, xi_inv, 1), (xi_inv, 1, xi),
            (1, xi, -xi_inv), (xi, -xi_inv, 1), (-xi_inv, 1, xi),
            (1, -xi, xi_inv), (-xi, xi_inv, 1), (xi_inv, 1, -xi),
            (1, -xi, -xi_inv), (-xi, -xi_inv, 1), (-xi_inv, 1, -xi),
            (-1, xi, xi_inv), (xi, xi_inv, -1), (xi_inv, -1, xi),
            (-1, xi, -xi_inv), (xi, -xi_inv, -1), (-xi_inv, -1, xi),
            (-1, -xi, xi_inv), (-xi, xi_inv, -1), (xi_inv, -1, -xi),
            (-1, -xi, -xi_inv), (-xi, -xi_inv, -1), (-xi_inv, -1, -xi),
        ]
        coords.extend(perms)

        vertices = list(range(24))
        edges = set()
        # Snub cube has exactly 60 edges - use precise distance-based selection
        # Calculate all distances and take the shortest 60
        all_distances = []
        for i in range(24):
            for j in range(i + 1, 24):
                dist2 = PolyhedronGenerators._dist2(coords[i], coords[j])
                dist = dist2 ** 0.5
                all_distances.append((dist, i, j))

        # Sort by distance and take the first 60 edges
        all_distances.sort()
        for dist, i, j in all_distances[:60]:
            edges.add((i, j))
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_stellated_tetrahedron(return_coords: bool = False):
        base_coords = [
            (1, 1, 1),
            (1, -1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
        ]
        faces = [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2],
        ]
        vertices, edges, coords = PolyhedronGenerators._stellate_polyhedron(
            base_coords, faces, height_factor=1.4
        )
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_stellated_cube(return_coords: bool = False):
        base_coords = [
            (-1, -1, -1), (1, -1, -1), (-1, 1, -1), (1, 1, -1),
            (-1, -1, 1), (1, -1, 1), (-1, 1, 1), (1, 1, 1),
        ]
        faces = [
            [0, 1, 3, 2],  # bottom (z = -1)
            [4, 5, 7, 6],  # top (z = 1)
            [0, 1, 5, 4],  # front (y = -1)
            [2, 3, 7, 6],  # back (y = 1)
            [0, 2, 6, 4],  # left (x = -1)
            [1, 3, 7, 5],  # right (x = 1)
        ]
        vertices, edges, coords = PolyhedronGenerators._stellate_polyhedron(
            base_coords, faces, height_factor=1.2
        )
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_stellated_octahedron(return_coords: bool = False):
        base_coords = [
            (0, 0, 1),
            (0, 0, -1),
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
        ]
        faces = [
            [0, 2, 5],
            [0, 5, 3],
            [0, 3, 4],
            [0, 4, 2],
            [1, 5, 2],
            [1, 3, 5],
            [1, 4, 3],
            [1, 2, 4],
        ]
        vertices, edges, coords = PolyhedronGenerators._stellate_polyhedron(
            base_coords, faces, height_factor=1.0
        )
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def undirected_stellated_triangular_prism(return_coords: bool = False):
        s3 = sqrt(3) / 2
        base_coords = [
            (1, 0, 1),
            (-0.5, s3, 1),
            (-0.5, -s3, 1),
            (1, 0, -1),
            (-0.5, s3, -1),
            (-0.5, -s3, -1),
        ]
        faces = [
            [0, 1, 2],
            [5, 4, 3],
            [0, 1, 4, 3],
            [1, 2, 5, 4],
            [2, 0, 3, 5],
        ]
        vertices, edges, coords = PolyhedronGenerators._stellate_polyhedron(
            base_coords, faces, height_factor=1.1
        )
        return PolyhedronGenerators._finalize(vertices, edges, coords, return_coords)

    @staticmethod
    def _dist2(p, q):
        return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2

    @staticmethod
    def _stellate_polyhedron(base_coords, faces, height_factor):
        coords = list(base_coords)
        edges = set()

        for face in faces:
            if len(face) < 3:
                raise ValueError("Faces must have at least three vertices for stellation")

            # Add the base edges along the face boundary
            face_len = len(face)
            for idx in range(face_len):
                u = face[idx]
                v = face[(idx + 1) % face_len]
                if u == v:
                    continue
                edges.add((min(u, v), max(u, v)))

            # Compute centroid of the face
            centroid = tuple(
                sum(coords[vertex][axis] for vertex in face) / face_len
                for axis in range(3)
            )

            # Determine outward normal direction
            a = coords[face[0]]
            b = coords[face[1]]
            c = coords[face[2]]
            normal = PolyhedronGenerators._cross(
                PolyhedronGenerators._subtract(b, a),
                PolyhedronGenerators._subtract(c, a),
            )
            normal_length = PolyhedronGenerators._norm(normal)
            if normal_length < 1e-12:
                raise ValueError("Degenerate face encountered during stellation")
            direction = tuple(component / normal_length for component in normal)
            if PolyhedronGenerators._dot(direction, centroid) < 0:
                direction = tuple(-component for component in direction)

            # Scale spike relative to average distance from centroid to face vertices
            radius = sum(
                PolyhedronGenerators._dist2(coords[vertex], centroid) ** 0.5
                for vertex in face
            ) / face_len
            spike_vector = tuple(component * radius * height_factor for component in direction)
            apex = tuple(centroid[axis] + spike_vector[axis] for axis in range(3))

            apex_index = len(coords)
            coords.append(apex)
            for vertex in face:
                edges.add((min(vertex, apex_index), max(vertex, apex_index)))

        vertices = list(range(len(coords)))
        return vertices, edges, coords

    @staticmethod
    def _subtract(p, q):
        return (p[0] - q[0], p[1] - q[1], p[2] - q[2])

    @staticmethod
    def _cross(u, v):
        return (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )

    @staticmethod
    def _norm(vector):
        return sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)

    @staticmethod
    def _dot(u, v):
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


__all__ = ["PolyhedronGenerators"]
