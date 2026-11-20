r"""
The functions of the package, whose parameters and values follow those of
the ``sieve_integral`` library.

EXAMPLES::

    sage: # users should write the simpler: from sieve_integral_brion import sieve_integral
    sage: from sieve_integral_brion.api import sieve_integral
    sage: u, v = var("u, v")
    sage: sieve_integral([u + v == 1, 1/4 < u, u < v]).overlaps(RealBallField(100)(3).log())
    True

AUTHORS:

- Sary Drappeau

AI DISCLOSURE:

Claude Opus 5.5 wrote this module.
"""
from sage.misc.misc_c import prod
from sage.rings.real_arb import RealBallField

from .evaluate import evaluate_piece
from .inputs import as_polytope, check_positive_orthant, is_polytope_on_sum1
from .pieces import Trivial, prepare_piece


def sieve_integral(polytope_data, precision=20, facteur=None, processes=None,
                   verbose=0, engine=None):
    r"""
    Return a RealBall containing `\int_P dt_1 \cdots dt_k/(t_1 \cdots t_d)`.

    The polytope `P \subset \RR_{>0}^d` is either full-dimensional (`k =
    d`), or on the hyperplane `\sum_j t_j = 1` (`k = d - 1`).

    INPUT:

    - ``polytope_data`` -- a polytope, or an iterable of symbolic linear
      equalities and inequalities.

    - ``precision`` (default: 20) -- the integer `p`: the error bound is at
      most `2^{1-p}`.

    - ``facteur`` (default: ``None``) -- ``None``: for now, the polytope is
      not cut into pieces, unlike to the ``sieve_integral`` library.

    - ``processes`` (default: ``None``) -- ignored for now: the computation
      is not parallelized.

    - ``verbose`` (default: 0) -- if positive, print the degree of the rule,
      its number of nodes and the error bound, then the value.

    - ``engine`` (default: ``None``) -- ignored for now.

    OUTPUT: a ball of ``RealBallField(precision)``.

    EXAMPLES:

    On a rectangle, it is a product of logarithms::

        sage: from sieve_integral_brion.api import sieve_integral
        sage: box = Polyhedron(vertices=[(1, 1), (2, 1), (1, 3), (2, 3)])
        sage: value = sieve_integral(box, precision=50)
        sage: R = RealBallField(100)
        sage: value.overlaps(R(2).log() * R(3).log()), value.rad() < 2^-49
        (True, True)

    On a polytope of `t_1 + t_2 + t_3 = 1`, the integral is `\frac12
    \int_1^5 \log(6 - x) \, dx/x`::

        sage: polytope = Polyhedron(eqns=[(-1, 1, 1, 1)],
        ....:                       ieqs=[(-1/7, 1, 0, 0), (0, -1, 1, 0), (0, 0, -1, 1)])
        sage: value = sieve_integral(polytope, precision=50)
        sage: C = ComplexBallField(100)
        sage: value.overlaps(C.integral(lambda x, _: (6 - x).log() / x, 1, 5).real() / 2)
        True

    The polytope has to lie in the open positive orthant::

        sage: u, v = var("u, v")
        sage: sieve_integral([u + v == 1, 0 < u, u < v])
        Traceback (most recent call last):
        ...
        ValueError: The polytope must lie in the open positive orthant, which
        is a hypothesis of the method, but its coordinate number 0 goes down
        to 0.
    """
    if facteur is not None:
        # Cutting the polytope into pieces is planned for a later commit.
        raise NotImplementedError("only facteur=None is supported")
    polytope, _ = as_polytope(polytope_data)
    field = RealBallField(precision)
    if polytope.is_empty():
        return field.zero()
    check_positive_orthant(polytope)
    border = is_polytope_on_sum1(polytope)
    k = polytope.ambient_dimension() - border
    if k <= 1:
        if k == 0:              # the point t_1 = 1, or the point of RR^0
            return field.one() if border else field.zero()
        mini, maxi = polytope.bounding_box()
        a, b = mini[0], maxi[0]
        ratio = b * (1 - a) / (a * (1 - b)) if border else b / a
        # arb.log tends to deteriorate radius, so we raise precision for safety
        return field(RealBallField(precision + 64)(ratio).log())
    piece = prepare_piece(polytope, border, precision)
    if verbose > 0:
        if isinstance(piece, Trivial):
            print(f"sieve_integral: the trivial bound {float(piece.centre + piece.radius):.3g}")
        else:
            print(f"sieve_integral: degree {piece.degree}, {prod(piece.ms)} nodes, "
                  f"error bound {float(piece.error):.3g}")
    value = evaluate_piece(piece, field)
    if verbose > 0:
        print(f"sieve_integral returning {value}")
    return value
