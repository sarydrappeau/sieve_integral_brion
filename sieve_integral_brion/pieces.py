r"""
Preparation of a piece of a polytope for the radial Brion sum: rescaling
of its coordinates, degree of the rule and bound on its error (or its
trivial bound).

The coordinates of the piece `P` are divided by the centres `c_j = (t_j^-
+ t_j^+)/2` of their ranges `[t_j^-, t_j^+]` on `P`, so that `t_j \in [1 -
\rho_j, 1 + \rho_j]` on the rescaled piece `P'`, with `\rho_j = (t_j^+ -
t_j^-)/(t_j^+ + t_j^-)`. The integral of `dt_1 \cdots dt_k/(t_1 \cdots
t_d)` over `P` is then `\kappa \int_\Delta G`, where `\kappa = 1` if `P` is
full-dimensional and `\kappa = 1/c_d` if it lies on `\sum_j t_j = 1`
(Section 3.2 of ``mathematics/radial-brion.pdf``).

Let `\tau` be the rate of
:func:`~sieve_integral_brion.chebyshev.chebyshev_inverse_rate` for `\rho =
\max_j \rho_j`, and `E^{(e)}_M` the `\ell^1` tail beyond the degree `M` of
the Chebyshev series of `(1 + \eta u)^{-e}`, `\eta = 2\tau/(1 + \tau^2)`
(Section 2.5 ibid.). A positive rule exact in degree `N \geq D`
computes `\int_\Delta G` within

.. MATH::

    2 \operatorname{vol}(P') \sum_{n=0}^D w_n E^{(d+n)}_{N-n},

with `w = (1)`, `D = 0`, for the plain integrand (Theorem 3.4 there).

Each piece aims at an error at most `b = 2^{-p}` (Section 4.5 there):
``SHARE_TRUNCATION`` `\cdot b` for this bound, which determines `N`, and a
target of ``SHARE_EVALUATION`` `\cdot b` for the radii of the rescaled values at the
nodes. If the trivial bound `\kappa \operatorname{vol}(P') / \prod_j (1 - \rho_j)`
is below ``NEGLIGIBLE`` `\cdot b`, we replace the piece by an instance of
:class:`Trivial` recording the trivial bound.

EXAMPLES:

An integral from Chen's second example (Section 2.5.1 of
arXiv:2606.30428), on `P = \{1 < t_3 < t_2 - 1 < 3 - 10 t_1 < 2\}`, at 30
bits. The coordinates range over `[1/10, 1/5]`, `[2, 3]` and `[1, 2]`, so
that `\rho = (1/3, 1/5, 1/3)`, and the rate is `\tau = 3 - 2\sqrt 2`
rounded up to `8` bits. The volume of `P'` is that of `P` divided by the
product of the centres::

    sage: from sieve_integral_brion.pieces import prepare_piece
    sage: P = Polyhedron(ieqs=[(-1, 0, 0, 1), (-1, 0, 1, -1), (4, -10, -1, 0), (-1, 10, 0, 0)])
    sage: piece = prepare_piece(P, False, 30)
    sage: piece.centres, piece.rate
    ((3/20, 5/2, 3/2), 11/64)
    sage: piece.volume == P.volume() / prod(piece.centres)
    True
    sage: piece.degree == 2 * piece.ms[0] - 1, piece.error <= 59/64 * 2^-30
    (True, True)

A thin box is negligible: its trivial bound is `2^{-40}`::

    sage: thin = Polyhedron(vertices=[(1, 1), (1 + 2^-40, 1), (1, 2), (1 + 2^-40, 2)])
    sage: prepare_piece(thin, False, 20)
    Trivial(centre=1/2199023255552, radius=1/2199023255552)

AUTHORS:

- Sary Drappeau

AI DISCLOSURE:

Claude Opus 5.5 wrote this module, from a prototype it had written for the
``sieve_integral`` library.
"""
from dataclasses import dataclass
from functools import lru_cache

from sage.arith.misc import factorial
from sage.misc.misc_c import prod
from sage.rings.rational_field import QQ

from .chebyshev import chebyshev_inverse_powers, chebyshev_inverse_rate
from .geometry import brion_volume, generator_table, vertex_cones

# the shares of the budget b of a piece: the Chebyshev coefficients of the
# weights left out (this is not implemented yet), and the target for the radii
# of the values at the nodes; the truncation bound takes the rest
SHARE_COEFFICIENTS = QQ(1) / 16
SHARE_EVALUATION = QQ(1) / 64
SHARE_TRUNCATION = 1 - SHARE_COEFFICIENTS - SHARE_EVALUATION
# a piece whose trivial bound is below this fraction of b is replaced by it
NEGLIGIBLE = QQ(2)**-10
# the binary digits of the rate tau
RATE_BITS = 8
# the highest degree of the rule
MAX_DEGREE = 2047


@dataclass(frozen=True)
class Trivial:
    r"""
    A piece replaced by its trivial bound: its integral lies within
    ``radius`` of ``centre``, two rational numbers.
    """
    centre: object
    radius: object


@dataclass(frozen=True)
class PreparedPiece:
    r"""
    A piece prepared for the radial Brion sum.

    - ``centres`` -- the `c_j`.

    - ``shift`` -- `\sigma`, `0` if the piece is full-dimensional, `1` if it
      lies on `\sum_j t_j = 1`.

    - ``prefactor`` -- `\kappa`.

    - ``volume`` -- `\operatorname{vol}(P')`.

    - ``rate`` -- `\tau`.

    - ``vertices``, ``generators``, ``cones`` -- the data of
      :class:`~sieve_integral_brion._brion_sum.BrionSum` for `P'`
      (:func:`~sieve_integral_brion.geometry.generator_table`).

    - ``degree`` -- the degree `N` of the rule.

    - ``ms`` -- the numbers of nodes of the rule in each direction.

    - ``error`` -- the bound on the error of `\kappa` times the rule, a
      rational number.

    - ``tolerance_exponent`` -- the integer `e`: the values at the nodes
      are to have radii at most `2^e`.

    - ``precision`` -- the integer `p` of the budget `2^{-p}`.
    """
    centres: tuple
    shift: int
    prefactor: object
    volume: object
    rate: object
    vertices: tuple
    generators: tuple
    cones: tuple
    degree: int
    ms: tuple
    error: object
    tolerance_exponent: int
    precision: int


@lru_cache(maxsize=None)
def inverse_power_tails(rate, power, size):
    r"""
    Return the tails `E^{(e)}_M` for `e \leq` ``power`` and `-1 \leq M <`
    ``size``, see Section 2.5 of ``mathematics/radial-brion.pdf``. In
    the returned tuple of tails, the value `E^{(e)}_M` is located at
    the position `M + 1` of the row `e` (``tails[e][M + 1]``).

    `E^{(e)}_{-1}` is the whole mass `(1 - \eta)^{-e}`. The computation is
    exact (:func:`~sieve_integral_brion.chebyshev.chebyshev_inverse_powers`).

    EXAMPLES:

    For `\tau = 1/3`, `E^{(1)}_M = (5/4) \, 3^{-M}`::

        sage: from sieve_integral_brion.pieces import inverse_power_tails
        sage: inverse_power_tails(1/3, 1, 4)[1]
        (5/2, 5/4, 5/12, 5/36, 5/108)
    """
    coefficients = chebyshev_inverse_powers(rate, power, size)
    mass = (1 + rate**2) / (1 - rate)**2
    tails = []
    for e in range(power + 1):
        row = [mass**e]
        for c in coefficients[e]:
            row.append(row[-1] - abs(c))
        tails.append(tuple(row))
    return tuple(tails)


def truncation_bound(d, rate, N, weights):
    r"""
    Return `\sum_n w_n E^{(d+n)}_{N-n}`, for the weights `w_n` of
    ``weights``.

    See Theorem 3.4 of ``mathematics/radial-brion.pdf``.

    EXAMPLES::

        sage: from sieve_integral_brion.pieces import truncation_bound, inverse_power_tails
        sage: truncation_bound(1, 1/3, 2, [1]) == 5/4 * 3^-2
        True
        sage: tails = inverse_power_tails(1/3, 2, 32)
        sage: truncation_bound(1, 1/3, 2, [1, 3]) == tails[1][3] + 3 * tails[2][2]
        True
    """
    # a power of two of coefficients, for the cache
    size = 32
    while size <= N:
        size *= 2
    tails = inverse_power_tails(rate, d + len(weights) - 1, size)
    return sum((w * tails[d + n][max(N - n, -1) + 1] for n, w in enumerate(weights) if w),
               QQ(0))


def least_degree(d, rate, weights, target):
    r"""
    Return the least degree `N \geq D`, `D + 1` the length of ``weights``,
    whose :func:`truncation_bound` is at most ``target``.

    EXAMPLES:

    The least `N` with `(5/4) \, 3^{-N} \leq 10^{-3}`::

        sage: from sieve_integral_brion.pieces import least_degree
        sage: least_degree(1, 1/3, [1], 10^-3)
        7
        sage: 5/4 * 3^-7 <= 10^-3 < 5/4 * 3^-6
        True
    """
    N = len(weights) - 1
    while truncation_bound(d, rate, N, weights) > target:
        N += 1
        if N > MAX_DEGREE:
            raise ArithmeticError(f"no degree up to {MAX_DEGREE} has a truncation "
                                  f"bound below {target}")
    return N


def floor_log2(x):
    r"""
    Return the largest integer `e` with `2^e \leq x`, for a positive rational
    number `x`.

    EXAMPLES::

        sage: from sieve_integral_brion.pieces import floor_log2
        sage: floor_log2(1), floor_log2(3/4), floor_log2(2^-30), floor_log2(2^-30 * 3/2)
        (0, -1, -30, -30)
    """
    e = x.numerator().nbits() - x.denominator().nbits()
    return e if QQ(2)**e <= x else e - 1


def prepare_piece(polytope, border, precision):
    r"""
    Return the data of the radial Brion sum of ``polytope`` and the bound on
    its error, or its trivial bound if that is negligible.

    INPUT:

    - ``polytope`` -- a polytope of the open positive orthant, over ``QQ``,
      full-dimensional or on `\sum_j t_j = 1`.

    - ``border`` -- whether ``polytope`` lies on `\sum_j t_j = 1`.

    - ``precision`` -- the integer `p`: the error is to be at most `2^{-p}`.

    OUTPUT: a :class:`PreparedPiece`, or a :class:`Trivial` piece.

    EXAMPLES:

    The segment `t_1 + t_2 = 1`, `1/4 \leq t_1 \leq 1/2`, with `\kappa = 1/c_2`::

        sage: from sieve_integral_brion.pieces import prepare_piece
        sage: segment = Polyhedron(vertices=[(1/4, 3/4), (1/2, 1/2)])
        sage: piece = prepare_piece(segment, True, 20)
        sage: piece.shift, piece.prefactor, piece.generators, len(piece.ms)
        (1, 8/5, ((5, -3),), 1)
    """
    d = polytope.ambient_dimension()
    k = d - 1 if border else d
    mini, maxi = polytope.bounding_box()
    centres = tuple((QQ(a) + QQ(b)) / 2 for a, b in zip(mini, maxi))
    rhos = [(QQ(b) - QQ(a)) / (QQ(b) + QQ(a)) for a, b in zip(mini, maxi)]
    prefactor = 1 / centres[-1] if border else QQ(1)
    vertices, cones = vertex_cones(polytope, centres, border)
    volume = brion_volume(vertices, cones, k, check=True)
    budget = QQ(2)**-precision
    trivial = prefactor * volume / prod(1 - r for r in rhos)
    if trivial < NEGLIGIBLE * budget:
        return Trivial(trivial / 2, trivial / 2)
    rate = chebyshev_inverse_rate(max(rhos), RATE_BITS)
    scale = 2 * prefactor * volume
    N = least_degree(d, rate, [1], SHARE_TRUNCATION * budget / scale)
    # the rule with m points in each direction is exact in degree 2m - 1
    m = max(2, (N + 2) // 2)
    generators, indexed = generator_table(cones)
    # the rule's weights add up to 1/(d - 1)!
    tolerance = SHARE_EVALUATION * budget * factorial(d - 1) / prefactor
    return PreparedPiece(
        centres=centres, shift=d - k, prefactor=prefactor, volume=volume, rate=rate,
        vertices=tuple(vertices), generators=tuple(generators), cones=tuple(indexed),
        degree=2 * m - 1, ms=(m,) * (d - 1),
        error=scale * truncation_bound(d, rate, 2 * m - 1, [1]),
        tolerance_exponent=floor_log2(tolerance), precision=precision)
