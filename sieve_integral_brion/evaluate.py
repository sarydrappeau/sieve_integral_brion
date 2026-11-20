r"""
Evaluation of prepared pieces of polytopes: cubature by the
conical product rule (Stroud), using
:class:`~sieve_integral_brion._brion_sum.BrionSum` and taking into
acccount the prefactor, and error bound management.

The sum is computed at ``EXTRA_BITS`` bits above the precision of the
piece or of the result, whichever is larger, rounded up to a multiple
of `64`. Nodes whose values are not finite, or have too large radii
for the target precision, are computed again at twice the precision,
repeatedly and up to ``ESCALATIONS`` times, after which a node which
still fails raises ``ArithmeticError``.

EXAMPLES:

An integral from Chen's second example (Section 2.5.1 of
arXiv:2606.30428), on `P = \{1 < t_3 < t_2 - 1 < t_1 - 2 < 2\}`,
against PARI's numerical integration of `\int_3^4 \int_2^{t_1 - 1}
\log(t_2 - 1) \, \frac{dt_2}{t_2} \frac{dt_1}{t_1}`, in which the
integral over `t_3` is done::

    sage: from sieve_integral_brion.pieces import prepare_piece
    sage: from sieve_integral_brion.evaluate import evaluate_piece
    sage: P = Polyhedron(ieqs=[(-1, 0, 0, 1), (-1, 0, 1, -1), (-1, 1, -1, 0), (4, -1, 0, 0)])
    sage: value = evaluate_piece(prepare_piece(P, False, 40), RealBallField(40))
    sage: numerical = pari("intnum(t1 = 3, 4, intnum(t2 = 2, t1 - 1, log(t2 - 1)/t2)/t1)")
    sage: numerical
    0.0148863467273708
    sage: abs(value - RealBallField(53)(numerical)) < 2^-39, value.rad() < 2^-39
    (True, True)

AUTHORS:

- Sary Drappeau

AI DISCLOSURE:

Claude Opus 5.5 wrote this module.

"""
from ._brion_sum_loader import load_brion_sum
from .pieces import Trivial
from .rules import conical_factors

BrionSum = load_brion_sum().BrionSum

# EXTRA_BITS is added to the precision during the computation
# of the Brion sum, on top of the requested precision. This is due to
# the presence of a notable amount of cancellation between terms.
# The value 80 is based on the examples (a smaller value triggers
# recomputations of many note values that are too imprecise on the first pass;
# a larger value slows down the first pass).
EXTRA_BITS = 80
# the number of times we double the precision for the recomputation of
# nodes for which an earlier computation didn't give a precise
# enough value
ESCALATIONS = 4


def start_precision(piece, field):
    r"""
    Return the precision of the first evaluation of ``piece``, for a result
    in ``field``. The result is rounded up to the next multiple of 64, which
    costs almost nothing since mantissas of FLINT Arbs are stored on 64-bit limbs.

    EXAMPLES::

        sage: from sieve_integral_brion.pieces import prepare_piece
        sage: from sieve_integral_brion.evaluate import start_precision
        sage: box = Polyhedron(vertices=[(1, 1), (2, 1), (1, 3), (2, 3)])
        sage: piece = prepare_piece(box, False, 40)
        sage: start_precision(piece, RealBallField(40)), start_precision(piece, RealBallField(200))
        (128, 320)
    """
    bits = max(piece.precision, field.precision()) + EXTRA_BITS
    return 64 * ((bits + 63) // 64)


def brion_sum(piece, prec, ms=None):
    r"""
    Return the :class:`~sieve_integral_brion._brion_sum.BrionSum` of
    ``piece`` at the precision ``prec``, with the numbers of nodes ``ms``
    (by default, those of the piece).
    """
    return BrionSum(prec, conical_factors(piece.ms if ms is None else ms, prec),
                    piece.vertices, piece.generators, piece.cones, piece.shift,
                    tolerance_exponent=piece.tolerance_exponent)


def rule_sum(piece, prec, ms=None):
    r"""
    Return the prefactor of ``piece`` times the sum of the rule with
    ``ms`` nodes (by default, those of the piece) over its nodes, starting at
    the precision ``prec``.

    EXAMPLES:

    With a tolerance below the radii at the first precision, all the nodes
    are computed again::

        sage: import dataclasses
        sage: from sieve_integral_brion.pieces import prepare_piece
        sage: from sieve_integral_brion.evaluate import brion_sum, rule_sum
        sage: box = Polyhedron(vertices=[(1, 1), (2, 1), (1, 3), (2, 3)])
        sage: piece = prepare_piece(box, False, 30)
        sage: tight = dataclasses.replace(piece, tolerance_exponent=-150)
        sage: len(brion_sum(tight, 128).run()[1]) == piece.ms[0]
        True
        sage: rule_sum(tight, 128).overlaps(rule_sum(piece, 128))
        True

    A node on a hyperplane `\langle x, g\rangle = 0` fails at every
    precision: here the middle node of three, on the segment `t_1 + t_2 =
    1` along `e_1 - e_2` with equal centres::

        sage: segment = Polyhedron(vertices=[(1/4, 3/4), (3/4, 1/4)])
        sage: piece = dataclasses.replace(prepare_piece(segment, True, 30), ms=(3,))
        sage: rule_sum(piece, 128)
        Traceback (most recent call last):
        ...
        ArithmeticError: the integrand is not finite, or has a radius above
        2^-37, at the nodes [(1,)] of the rule with (3,) nodes, on the piece
        with vertices [...], even at 2048 bits
    """
    ms = piece.ms if ms is None else tuple(ms)
    total, failed = brion_sum(piece, prec, ms).run()
    for _ in range(ESCALATIONS):
        if not failed:
            break
        prec *= 2
        value, failed = brion_sum(piece, prec, ms).run(explicit=failed)
        total += value
    if failed:
        vertices = [tuple(c * x for c, x in zip(piece.centres, v)) for v in piece.vertices]
        raise ArithmeticError(f"the integrand is not finite, or has a radius above "
                              f"2^{piece.tolerance_exponent}, at the nodes {failed} of "
                              f"the rule with {ms} nodes, on the piece with vertices "
                              f"{vertices}, even at {prec} bits")
    return total * piece.prefactor


def evaluate_piece(piece, field):
    r"""
    Return a ball of ``field`` containing the integral over ``piece``, a
    :class:`~sieve_integral_brion.pieces.PreparedPiece` or a
    :class:`~sieve_integral_brion.pieces.Trivial` piece.

    EXAMPLES::

        sage: from sieve_integral_brion.pieces import Trivial
        sage: from sieve_integral_brion.evaluate import evaluate_piece
        sage: value = evaluate_piece(Trivial(1/2, 1/2), RealBallField(20))
        sage: value.contains_exact(0), value.contains_exact(1)
        (True, True)
    """
    if isinstance(piece, Trivial):
        return field(piece.centre).add_error(piece.radius)
    value = rule_sum(piece, start_precision(piece, field))
    return field(value).add_error(piece.error)
