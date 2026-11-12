r"""
Check that the vertex cones give the volumes of polytopes, simple or
not, full-dimensional or on the hyperplane `\sum_j t_j = 1`; check that
the grid lines follow their definition, and that the grid pieces are the
intersections of the polytope with the cells of the grid, one per cell,
which make up its volume.
"""
import random

from sieve_integral_brion import geometry as brion_geometry


def geometry_volume(polytope):
    r"""
    The volume of ``polytope``, measured by `dt_1 \cdots dt_{d-1}` if it lies
    on `\sum_j t_j = 1`.
    """
    if polytope.equations_list():
        return Polyhedron(vertices=[v[:-1] for v in polytope.vertices_list()]).volume()
    return polytope.volume()


def on_hyperplane(points):
    r"""
    The polytope spanned by the points `(p, 1 - \sum_j p_j)`.
    """
    return Polyhedron(vertices=[tuple(p) + (1 - sum(p),) for p in points], base_ring=QQ)


def random_polytope(rng, d, border):
    r"""
    The convex hull of `3d` random points, in `[1, 3]^d`, or on `\sum_j t_j
    = 1` with `t_j \geq 1/(5d)`.
    """
    if border:
        points = [[QQ(rng.randint(5, 15)) / (25 * d) for _ in range(d - 1)]
                  for _ in range(3 * d)]
        return on_hyperplane(points)
    return Polyhedron(vertices=[[1 + QQ(rng.randint(0, 14)) / 7 for _ in range(d)]
                                for _ in range(3 * d)], base_ring=QQ)


def geometry_polytopes():
    r"""
    Named polytopes, simple or not, full-dimensional or not.
    """
    rng = random.Random(int(2026))
    pyramid = [(1, 1, 1), (3, 1, 1), (1, 3, 1), (3, 3, 1), (2, 2, 4)]
    octahedron = [tuple(v) for v in polytopes.cross_polytope(3).vertices_list()]
    cases = {
        "segment": Polyhedron(vertices=[(1,), (3,)]),
        "border/segment": on_hyperplane([[1/5], [3/5]]),
        "octahedron": polytopes.cross_polytope(3) + vector((2, 3, 5)),
        "pyramid": Polyhedron(vertices=pyramid),
        "cross-polytope": polytopes.cross_polytope(4) + vector((2, 2, 3, 4)),
        "border/pyramid": on_hyperplane([[c / 20 for c in p] for p in pyramid]),
        "border/octahedron": on_hyperplane([[(c + 2) / 20 for c in p] for p in octahedron]),
        "border/simplex": vector((1/10,) * 4) + 6/10 * polytopes.simplex(3),
    }
    for d in (2, 3, 4):
        cases[f"random/d{d}"] = random_polytope(rng, d, False)
    for d in (3, 4, 5):
        cases[f"border/random/d{d}"] = random_polytope(rng, d, True)
    return cases


def test_integer_algebra():
    rng = random.Random(int(4))
    for n in range(1, 7):
        for _ in range(20):
            rows = [[rng.randint(-5, 5) for _ in range(n)] for _ in range(n)]
            if n > 1 and rng.random() < 0.3:
                rows[-1] = [a - 2 * b for a, b in zip(rows[0], rows[1])]
            assert brion_geometry.int_det(rows) == matrix(ZZ, rows).det(), rows
            wide = rows + [[rng.randint(-3, 3) for _ in range(n)]]
            assert brion_geometry.int_rank(wide) == matrix(ZZ, wide).rank(), wide
            entries = [QQ(rng.randint(-9, 9)) / rng.randint(1, 9) for _ in range(n)]
            if any(entries):
                ray = brion_geometry.primitive(entries)
                assert gcd(ray) == 1, ray
                i = next(i for i, c in enumerate(entries) if c)
                scale = entries[i] / ray[i]
                assert scale > 0 and vector(entries) == scale * vector(ray), (entries, ray)


def test_cone_volumes():
    rng = random.Random(int(7))
    for name, polytope in geometry_polytopes().items():
        border = bool(polytope.equations_list())
        d = polytope.ambient_dimension()
        k = d - 1 if border else d
        mini, maxi = polytope.bounding_box()
        boxed = tuple((a + b) / 2 for a, b in zip(mini, maxi))
        drawn = tuple(QQ(rng.randint(1, 30)) / rng.randint(1, 30) for _ in range(d))
        expected = geometry_volume(polytope)
        for centres in ((1,) * d, boxed, drawn):
            vertices, cones = brion_geometry.vertex_cones(polytope, centres, border)
            for i, rays, det in cones:
                assert len(rays) == k and det != 0, (name, i, rays)
                assert det == abs(matrix(ZZ, [ray[:k] for ray in rays]).det())
                assert all(gcd(ray) == 1 for ray in rays), (name, rays)
            volume = brion_geometry.brion_volume(vertices, cones, k, check=True)
            assert brion_geometry.brion_volume(vertices, cones, k) == volume, name
            scale = prod(QQ(c) for c in centres[:k])
            assert volume * scale == expected, (name, centres, volume * scale, expected)
        if not polytope.is_simple():
            assert len(cones) > len(vertices), name


def grid_cell(polytope, piece, facteur):
    r"""
    The cell of the grid of ``polytope`` which holds ``piece``, as the
    pairs of consecutive lines around each coordinate of the piece.
    """
    mini, maxi = polytope.bounding_box()
    low, high = piece.bounding_box()
    cell = []
    for j in range(polytope.ambient_dimension()):
        lines = [mini[j]] + brion_geometry.grid_lines(mini[j], maxi[j], facteur) + [maxi[j]]
        a = max(line for line in lines if line <= low[j])
        b = lines[lines.index(a) + 1]
        assert high[j] <= b, (j, low[j], high[j], lines)
        cell.append((a, b))
    return tuple(cell)


def grid_cases():
    rng = random.Random(int(11))
    return [
        (Polyhedron(ieqs=[(-1, 1, 0), (285605/100000, -1, 0), (-1, 0, 1), (6/5, 0, -1)]), 1.3),
        (polytopes.hypercube(3) + vector((4, 4, 4)), 1.3),
        (vector((1/10,) * 4) + 6/10 * polytopes.simplex(3), 2),
        (vector((1/10,) * 4) + 6/10 * polytopes.simplex(3), 3/2),
        (random_polytope(rng, 3, False), 1.6),
        (random_polytope(rng, 4, True), 2.5),
        (random_polytope(rng, 2, False), float(1.15)),
        (random_polytope(rng, 3, False), 4),
    ]


def test_grid_lines():
    step = brion_geometry.GRID_STEP
    for low, high, facteur in [(1, 3, 1.3), (3/7, 5, 1.15), (2, 2, 1.3), (1, 1000, 4),
                               (1/10, 7/10, 3/2), (5, 9, float(1.6)), (1, 10, 2.5),
                               (1, 101/100, 1.0002), (2/3, 17/25, 1.0001)]:
        lines = brion_geometry.grid_lines(QQ(low), QQ(high), facteur)
        ratio = QQ(RR(facteur))
        assert all(low < line < high for line in lines), (facteur, lines)
        assert all(a < b for a, b in zip(lines, lines[1:])), (facteur, lines)
        expected, n = set(), 1
        while low * (ratio^n - step) < high:
            nearest = (ratio^n / step).round() * step
            assert abs(nearest - ratio^n) <= step / 2
            if low < low * nearest < high:
                expected.add(low * nearest)
            n += 1
        assert lines == sorted(expected), (facteur, lines, sorted(expected))
    try:
        brion_geometry.grid_lines(1, 2, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("a ratio of 1 is accepted")


def test_grid_pieces():
    for polytope, facteur in grid_cases():
        border = bool(polytope.equations_list())
        k = polytope.dimension()
        d = polytope.ambient_dimension()
        pieces = brion_geometry.grid_pieces(polytope, facteur)
        cells = [grid_cell(polytope, piece, facteur) for piece in pieces]
        assert len(set(cells)) == len(cells), facteur
        for piece, cell in zip(pieces, cells):
            assert piece.dimension() == k, (facteur, cell)
            box = [[-a] + [int(i == j) for i in range(d)] for j, (a, _) in enumerate(cell)]
            box += [[b] + [-int(i == j) for i in range(d)] for j, (_, b) in enumerate(cell)]
            expected = Polyhedron(ieqs=polytope.inequalities_list() + box,
                                  eqns=polytope.equations_list(), base_ring=QQ)
            assert piece == expected, (facteur, cell)
        assert sum(geometry_volume(piece) for piece in pieces) == geometry_volume(polytope)
        for piece in pieces:
            vertices, cones = brion_geometry.vertex_cones(piece, (1,) * piece.ambient_dimension(),
                                                          border)
            assert brion_geometry.brion_volume(vertices, cones, k, check=True) == geometry_volume(piece)


TESTS = {
    "geometry/integer-algebra": (test_integer_algebra, ("quick",)),
    "geometry/cone-volumes": (test_cone_volumes, ("quick",)),
    "geometry/grid-lines": (test_grid_lines, ("quick",)),
    "geometry/grid-pieces": (test_grid_pieces, ("quick",)),
}
