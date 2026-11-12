r"""
Vertex cones of a polytope, and the grids which cut it into pieces.

A polytope `P \subset \RR_{>0}^d` is either full-dimensional (`k = d`), or
lies on the hyperplane `\sum_j t_j = 1` (`k = d - 1`), where it is measured
by `dt_1 \cdots dt_k`. Brion's formula reads, as an identity of
meromorphic functions of `x \in \CC^d`,

.. MATH::

    \int_P e^{-\langle x, t\rangle} dt_1 \cdots dt_k = \sum_C |\det C|
    \frac{e^{-\langle x, v_C\rangle}}{\prod_{g \in C} \langle x, g\rangle},

over simplicial cones `C = v_C + \operatorname{cone}(g)_{g \in C}` which
triangulate the vertex cones of `P`. Their rays `g` are primitive integer
vectors of `\RR^d`, and `\det C` is the determinant of their first `k`
coordinates.

The functions below apply it to the rescaled polytope `P' = \{s : (c_1
s_1, \ldots, c_d s_d) \in P\}`, for positive numbers `c_j` (later, the
centres of the ranges of the coordinates on `P`, so that `P'` lies around
`(1, \ldots, 1)`). When `k = d - 1`, `P'` lies on `\sum_j c_j s_j = 1` and
is measured by `ds_1 \cdots ds_k`: the formula holds for it as well, since
this hyperplane too projects injectively onto the first `k` coordinates.

The grid of ratio `f` cuts the `j`-th coordinate at the lines `m_j q_n`,
`n \geq 1`, where `m_j` is the least value of `t_j` on `P` and `q_n` the
multiple of `2^{-10}` nearest to `f^n` (the ``sieve_integral`` library
rounds `f^n` to three decimals instead). Each piece lies between two
consecutive lines in each coordinate.

EXAMPLES::

    sage: from sieve_integral_brion.geometry import vertex_cones, brion_volume
    sage: octahedron = polytopes.cross_polytope(3) + vector((2, 2, 2))
    sage: vertices, cones = vertex_cones(octahedron, (1, 1, 1), False)
    sage: len(vertices), len(cones)
    (6, 12)
    sage: brion_volume(vertices, cones, 3), octahedron.volume()
    (4/3, 4/3)

REFERENCES:

.. [Hud1969] J. F. P. Hudson, *Piecewise Linear Topology*, W. A. Benjamin,
   1969, Lemma 1.4.

AUTHORS:

- Sary Drappeau

AI DISCLOSURE:

Claude Opus 5.5 wrote this module, from a prototype it had written for the
``sieve_integral`` library.
"""
import math
import random

from sage.arith.misc import factorial
from sage.geometry.polyhedron.constructor import Polyhedron
from sage.misc.misc_c import prod
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.rings.real_mpfr import RR

# the lines of the grids are multiples of this step
GRID_STEP = QQ(2)**-10


def primitive(vector):
    r"""
    Return the positive multiple of the rational vector ``vector`` whose
    entries are coprime integers.

    EXAMPLES::

        sage: from sieve_integral_brion.geometry import primitive
        sage: primitive((2/3, -4, 0))
        (1, -6, 0)
    """
    vector = [QQ(c) for c in vector]
    scale = math.lcm(*(int(c.denominator()) for c in vector))
    entries = [int(c.numerator()) * (scale // int(c.denominator())) for c in vector]
    divisor = math.gcd(*entries)
    return tuple(ZZ(e // divisor) for e in entries)


def int_rank(rows):
    r"""
    Return the rank of a list of integer vectors, by fraction-free
    elimination.

    EXAMPLES::

        sage: from sieve_integral_brion.geometry import int_rank
        sage: int_rank([(1, 2, 3), (2, 4, 6), (0, 1, 1)])
        2
    """
    rows = [list(row) for row in rows]
    rank = 0
    for column in range(len(rows[0]) if rows else 0):
        pivot = next((r for r in range(rank, len(rows)) if rows[r][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        top = rows[rank]
        for r in range(rank + 1, len(rows)):
            if rows[r][column]:
                factor = rows[r][column]
                rows[r] = [top[column] * x - factor * y for x, y in zip(rows[r], top)]
        rank += 1
    return rank


def int_det(rows):
    r"""
    Return the determinant of a square integer matrix, given by its rows,
    by Bareiss's algorithm.

    EXAMPLES::

        sage: from sieve_integral_brion.geometry import int_det
        sage: int_det([(0, 2, 1), (1, 1, 0), (3, 0, 1)])
        -5
    """
    a = [[int(x) for x in row] for row in rows]
    n = len(a)
    sign, previous = 1, 1
    for k in range(n - 1):
        if a[k][k] == 0:
            swap = next((r for r in range(k + 1, n) if a[r][k]), None)
            if swap is None:
                return ZZ(0)
            a[k], a[swap] = a[swap], a[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) // previous
        previous = a[k][k]
    return ZZ(sign * a[n - 1][n - 1])


def pulling(rays, k, facets, rows, ranks):
    r"""
    Return the pulling triangulation of a `k`-dimensional cone: the least
    of its rays, joined to the triangulations of its facets which do not
    contain that ray. This is the construction of [Hud1969]_, Lemma 1.4,
    for the polytope cut out of the cone by a hyperplane, whose vertices are
    the rays.

    INPUT:

    - ``rays`` -- the frozen set of the indices of the rays of the cone.

    - ``k`` -- the dimension of the cone.

    - ``facets`` -- the facets of the cone of which this cone is a face,
      as sets of indices of rays. The facets of the face are its
      intersections of dimension `k - 1` with them.

    - ``rows`` -- the rays, integer vectors indexed as in ``rays``, of
      length the dimension of the whole cone.

    - ``ranks`` -- a dictionary, a cache of the ranks of the faces.

    OUTPUT: a list of frozen sets of `k` indices.

    EXAMPLES:

    The cone over a square splits into two simplicial cones::

        sage: from sieve_integral_brion.geometry import pulling
        sage: rows = [(1, 0, 1), (0, 1, 1), (-1, 0, 1), (0, -1, 1)]
        sage: facets = [frozenset({0, 1}), frozenset({1, 2}),
        ....:           frozenset({2, 3}), frozenset({3, 0})]
        sage: sorted(sorted(s) for s in pulling(frozenset(range(4)), 3, facets, rows, {}))
        [[0, 1, 2], [0, 2, 3]]
    """
    if len(rays) == k:
        return [rays]
    first = min(rays)
    seen, simplices = set(), []
    for facet in facets:
        face = facet & rays
        if first in face or face in seen or len(face) < k - 1:
            continue
        seen.add(face)
        if face not in ranks:
            ranks[face] = int_rank([rows[n] for n in face])
        if ranks[face] != k - 1:
            continue
        simplices += [s | {first} for s in pulling(face, k - 1, facets, rows, ranks)]
    return simplices


def vertex_cones(polytope, centres, border):
    r"""
    Return the vertices of the rescaled polytope `P'` (``polytope`` with its
    coordinates divided by ``centres``), and the simplicial cones which
    triangulate its vertex cones.

    INPUT:

    - ``polytope`` -- a polytope `P`, full-dimensional or on `\sum_j t_j =
      1`.

    - ``centres`` -- the `d` positive rational numbers `c_j`.

    - ``border`` -- whether ``polytope`` lies on `\sum_j t_j = 1`.

    OUTPUT: a pair ``(vertices, cones)``: the vertices of `P'`, tuples of
    rational numbers, and triples ``(index, rays, det)``, where ``index`` is
    the position of the vertex of the cone, ``rays`` its `k` rays, primitive
    integer vectors of length `d` along edges of `P'`, and ``det`` the
    absolute value of the determinant of their first `k` coordinates. The
    vertex cones with more than `k` rays are triangulated by
    :func:`pulling`.

    EXAMPLES:

    The rectangle `[1, 2] \times [1, 3]`, with centres `(3/2, 2)`::

        sage: from sieve_integral_brion.geometry import vertex_cones
        sage: box = Polyhedron(vertices=[(1, 1), (2, 1), (1, 3), (2, 3)])
        sage: vertices, cones = vertex_cones(box, (3/2, 2), False)
        sage: vertices
        [(2/3, 1/2), (2/3, 3/2), (4/3, 1/2), (4/3, 3/2)]
        sage: cones
        [(0, ((0, 1), (1, 0)), 1),
         (1, ((0, -1), (1, 0)), 1),
         (2, ((-1, 0), (0, 1)), 1),
         (3, ((-1, 0), (0, -1)), 1)]

    A triangle on `t_1 + t_2 + t_3 = 1`, whose rays have three
    coordinates::

        sage: triangle = Polyhedron(vertices=[(1/2, 1/4, 1/4), (1/4, 1/2, 1/4),
        ....:                                 (1/4, 1/4, 1/2)])
        sage: vertices, cones = vertex_cones(triangle, (1, 1, 1), True)
        sage: cones[0]
        (0, ((0, 1, -1), (1, 0, -1)), 1)
    """
    d = polytope.ambient_dimension()
    k = d - 1 if border else d
    vertices = [tuple(QQ(c) / s for c, s in zip(v, centres))
                for v in polytope.vertices_list()]
    adjacency = polytope.vertex_adjacency_matrix()
    incidence = polytope.incidence_matrix()
    inequalities = [h.index() for h in polytope.inequality_generator()]
    cones = []
    for i, vertex in enumerate(vertices):
        neighbours = adjacency.nonzero_positions_in_row(i)
        rays = [primitive([b - a for a, b in zip(vertex, vertices[j])])
                for j in neighbours]
        if len(rays) == k:
            simplices = [range(k)]
        else:
            # the facets of the vertex cone: for each inequality tight at
            # the vertex, the rays of the edges on it
            facets = {frozenset(n for n, j in enumerate(neighbours) if incidence[j, h])
                      for h in inequalities if incidence[i, h]}
            rows = [ray[:k] for ray in rays]
            simplices = pulling(frozenset(range(len(rays))), k, facets, rows, {})
        for simplex in simplices:
            simplex_rays = tuple(rays[n] for n in sorted(simplex))
            cones.append((i, simplex_rays,
                          abs(int_det([ray[:k] for ray in simplex_rays]))))
    return vertices, cones


def brion_volume(vertices, cones, k, check=False, seed=1):
    r"""
    Return the volume of the rescaled polytope `P'` from its vertex cones,
    exactly.

    It is the term of degree `0` in `z` of `\sum_C |\det C| e^{-z
    \langle\lambda, v_C\rangle} / \prod_{g \in C} z \langle\lambda,
    g\rangle`, for an integer vector `\lambda` of length `k` off every
    hyperplane `\langle\lambda, g\rangle = 0`. The terms of negative degree
    vanish, by Brion's formula, which gives a test of the cones.

    INPUT:

    - ``vertices``, ``cones`` -- as returned by :func:`vertex_cones`.

    - ``k`` -- the dimension of the polytope.

    - ``check`` (default: ``False``) -- whether to check that the terms of
      negative degree vanish, and raise ``ArithmeticError`` otherwise.

    - ``seed`` (default: 1) -- the seed of the draw of `\lambda`.

    OUTPUT: the volume of `P'`, measured by `ds_1 \cdots ds_k`; that of `P`
    is `c_1 \cdots c_k` times as large.

    EXAMPLES::

        sage: from sieve_integral_brion.geometry import vertex_cones, brion_volume
        sage: box = Polyhedron(vertices=[(1, 1), (2, 1), (1, 3), (2, 3)])
        sage: brion_volume(*vertex_cones(box, (3/2, 2), False), 2)
        2/3

    The cones of a single vertex fail the test::

        sage: vertices, cones = vertex_cones(box, (1, 1), False)
        sage: brion_volume(vertices, cones[:1], 2, check=True)
        Traceback (most recent call last):
        ...
        ArithmeticError: the cones fail Brion's formula
    """
    rng = random.Random(seed)
    rays = {ray for _, simplex_rays, _ in cones for ray in simplex_rays}
    while True:
        lam = [rng.randint(-1000, 1000) for _ in range(k)]
        if all(sum(a * b for a, b in zip(lam, ray)) for ray in rays):
            break
    volume = QQ(0)
    negative = [QQ(0)] * k
    for i, simplex_rays, det in cones:
        ell = sum(a * b for a, b in zip(lam, vertices[i]))
        term = QQ(det) / prod(sum(a * b for a, b in zip(lam, ray)) for ray in simplex_rays)
        if check:
            for n in range(k):
                negative[n] += term
                term *= -ell
        else:
            term *= (-ell)**k
        volume += term
    if any(negative):
        raise ArithmeticError("the cones fail Brion's formula")
    return volume / factorial(k)


def grid_lines(low, high, facteur):
    r"""
    Return the values `low \cdot q_n`, `n \geq 1`, strictly between ``low``
    and ``high``, where `q_n` is the multiple of ``GRID_STEP`` `= 2^{-10}`
    nearest to `f^n`.

    INPUT:

    - ``low``, ``high`` -- positive rational numbers.

    - ``facteur`` -- a real number `f > 1`, read as a rational number
      (``1.3`` as `13/10`).

    OUTPUT: the increasing list of the lines, meaning the axis values that will be used to cut the polytopes.

    EXAMPLES:

    The multiples of `2^{-10}` nearest to `1.3^n`, `n \geq 1` lying below `3`
    the first one lies below `1.3`::

        sage: from sieve_integral_brion.geometry import grid_lines
        sage: grid_lines(1, 3, 1.3)
        [1331/1024, 1731/1024, 1125/512, 2925/1024]
    """
    ratio = QQ(RR(facteur))
    if ratio <= 1:
        raise ValueError("The balancing parameter must be greater than 1.")
    lines = []
    n = 1
    while True:
        line = low * (ratio**n / GRID_STEP).round() * GRID_STEP
        if line >= high:
            return lines
        if line > (lines[-1] if lines else low):
            lines.append(line)
        n += 1


def grid_pieces(polytope, facteur):
    r"""
    Return the pieces of ``polytope`` cut by the grid of ratio ``facteur``
    which starts at its lower corner, one coordinate after another.

    The pieces of lower dimension are dropped.

    INPUT:

    - ``polytope`` -- a polytope with positive coordinates.

    - ``facteur`` -- a real number `f > 1`.

    OUTPUT: a list of polytopes.

    EXAMPLES:

    The grid of ratio `1.3` on `[3, 5]^3` makes eight pieces::

        sage: from sieve_integral_brion.geometry import grid_pieces
        sage: cube = polytopes.hypercube(3) + vector((4, 4, 4))
        sage: pieces = grid_pieces(cube, 1.3)
        sage: len(pieces), sum(piece.volume() for piece in pieces)
        (8, 8)
    """
    d = polytope.ambient_dimension()
    dimension = polytope.dimension()
    mini, maxi = polytope.bounding_box()
    equations = polytope.equations_list()
    pieces = [polytope]
    for j in range(d):
        lines = grid_lines(QQ(mini[j]), QQ(maxi[j]), facteur)
        if not lines:
            continue
        cut = []
        for piece in pieces:
            column = [QQ(v[j]) for v in piece.vertices_list()]
            low, high = min(column), max(column)
            cuts = [line for line in lines if low < line < high]
            if not cuts:
                cut.append(piece)
                continue
            inequalities = piece.inequalities_list()
            for a, b in zip([low] + cuts, cuts + [high]):
                above = [-a] + [int(i == j) for i in range(d)]
                below = [b] + [-int(i == j) for i in range(d)]
                part = Polyhedron(ieqs=inequalities + [above, below],
                                  eqns=equations, base_ring=QQ)
                if part.dimension() == dimension:
                    cut.append(part)
        pieces = cut
    return pieces
