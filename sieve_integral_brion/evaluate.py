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

A node which still fails at ``SINGULAR_FACTOR`` times the first precision,
while some `\langle x, g\rangle` contains `0`, is *singular*: it lies on a
hyperplane where the terms of the radial Brion formula have poles, and
raises :class:`SingularNode`. :func:`integrate_piece` then computes the
whole piece again with another configuration of the rule
(:func:`~sieve_integral_brion.rules.next_config`), and if that fails, it
tries perturbed centres (:func:`~sieve_integral_brion.pieces.perturbed_centres`),
up to ``ATTEMPTS`` attempts in all. The random choices are seeded by the
vertices of the piece (:func:`piece_rng`).

The dictionary ``counts`` holds the numbers of nodes summed, of those
computed again, of singular nodes, and of reconfigurations.

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
import random

from sage.misc.misc_c import prod

from ._brion_sum_loader import load_brion_sum
from .geometry import bad_pairs, primitive
from .pieces import Trivial, perturbed_centres, prepare_piece
from .rules import conical_factors, next_config

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

# a node failing at this many times the first precision, on a hyperplane
# <x, g> = 0, is singular
SINGULAR_FACTOR = 4

# reconfigurations of the rule tried on a piece with a singular node
ATTEMPTS = 4

# since the module was loaded: the numbers of nodes summed by rule_sum, of
# those computed again, of singular nodes, and of reconfigurations
counts = {"nodes": 0, "retried": 0, "singular": 0, "reconfigured": 0}


class SingularNode(ArithmeticError):
    r"""
    Nodes of the rule lie on hyperplanes `\langle x, g\rangle = 0`.

    INPUT:

    - ``piece`` -- the :class:`~sieve_integral_brion.pieces.PreparedPiece`.

    - ``config`` -- the :class:`~sieve_integral_brion.rules.RuleConfig` of
      the rule.

    - ``nodes`` -- a dictionary: the multi-index of each singular node, and
      the positions of its generators `g` among those of the piece.
    """
    def __init__(self, piece, config, nodes):
        self.piece, self.config, self.nodes = piece, config, nodes
        node, positions = next(iter(nodes.items()))
        # the generators, as directions of edges of the polytope before rescaling
        edges = [primitive([c * x for c, x in zip(piece.centres, piece.generators[g])])
                 for g in positions]
        vertices = [tuple(c * x for c, x in zip(piece.centres, v)) for v in piece.vertices]
        message = (f"the node {node} of the rule {config} lies on hyperplanes "
                   f"<x, g> = 0, g along the edges {edges}")
        if len(nodes) > 1:
            message += f", and {len(nodes) - 1} other nodes on such hyperplanes"
        super().__init__(f"{message}, on the piece with vertices {vertices}")


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


def brion_sum(piece, prec, config=None):
    r"""
    Return the :class:`~sieve_integral_brion._brion_sum.BrionSum` of
    ``piece`` at the precision ``prec``, with the rule of ``config`` (by
    default, that of the piece).
    """
    config = piece.config if config is None else config
    return BrionSum(prec, conical_factors(config.ms, prec),
                    piece.vertices, piece.generators, piece.cones, piece.shift,
                    order=config.order, exponents=piece.exponents, slopes=piece.slopes,
                    derivatives=piece.derivatives, ratio=piece.ratio,
                    tolerance_exponent=piece.tolerance_exponent)


def rule_sum(piece, prec, config=None):
    r"""
    Return the prefactor of ``piece`` times the sum of the rule of
    ``config`` (by default, that of the piece) over its nodes, starting at
    the precision ``prec``.

    EXAMPLES:

    With a tolerance below the radii at the first precision, all the nodes
    are computed again::

        sage: import dataclasses
        sage: from sieve_integral_brion.pieces import prepare_piece
        sage: from sieve_integral_brion.evaluate import brion_sum, counts, rule_sum
        sage: box = Polyhedron(vertices=[(1, 1), (2, 1), (1, 3), (2, 3)])
        sage: piece = prepare_piece(box, False, 30)
        sage: tight = dataclasses.replace(piece, tolerance_exponent=-150)
        sage: len(brion_sum(tight, 128).run()[1]) == prod(piece.config.ms)
        True
        sage: retried = counts["retried"]
        sage: rule_sum(tight, 128).overlaps(rule_sum(piece, 128))
        True
        sage: counts["retried"] - retried == prod(piece.config.ms)
        True

    A node on a hyperplane `\langle x, g\rangle = 0` fails at every
    precision: here the middle node of three, on the segment `t_1 + t_2 =
    1` along `e_1 - e_2` with equal centres::

        sage: from sieve_integral_brion.rules import RuleConfig
        sage: segment = Polyhedron(vertices=[(1/4, 3/4), (3/4, 1/4)])
        sage: piece = prepare_piece(segment, True, 30)
        sage: rule_sum(piece, 128, RuleConfig((0, 1), (3,)))
        Traceback (most recent call last):
        ...
        SingularNode: the node (1,) of the rule RuleConfig(order=(0, 1), ms=(3,))
        lies on hyperplanes <x, g> = 0, g along the edges [(1, -1)], on the
        piece with vertices [(1/4, 3/4), (3/4, 1/4)]
    """
    config = piece.config if config is None else config
    total, failed = brion_sum(piece, prec, config).run()
    counts["nodes"] += prod(config.ms)
    singular_prec = SINGULAR_FACTOR * prec
    for _ in range(ESCALATIONS):
        if not failed:
            break
        counts["retried"] += len(failed)
        prec *= 2
        summer = brion_sum(piece, prec, config)
        value, failed = summer.run(explicit=failed)
        total += value
        if prec == singular_prec:
            singular = {node: summer.zero_generators(node) for node in failed}
            singular = {node: g for node, g in singular.items() if g}
            if singular:
                counts["singular"] += len(singular)
                raise SingularNode(piece, config, singular)
    if failed:
        vertices = [tuple(c * x for c, x in zip(piece.centres, v)) for v in piece.vertices]
        raise ArithmeticError(f"the integrand is not finite, or has a radius above "
                              f"2^{piece.tolerance_exponent}, at the nodes {failed} of "
                              f"the rule {config}, on the piece with vertices "
                              f"{vertices}, even at {prec} bits")
    return total * piece.prefactor


def evaluate_piece(piece, field, config=None):
    r"""
    Return a ball of ``field`` containing the integral over ``piece``, a
    :class:`~sieve_integral_brion.pieces.PreparedPiece` or a
    :class:`~sieve_integral_brion.pieces.Trivial` piece, with the rule of
    ``config`` (by default, that of the piece).

    EXAMPLES::

        sage: from sieve_integral_brion.pieces import Trivial
        sage: from sieve_integral_brion.evaluate import evaluate_piece
        sage: value = evaluate_piece(Trivial(1/2, 1/2), RealBallField(20))
        sage: value.contains_exact(0), value.contains_exact(1)
        (True, True)
    """
    if isinstance(piece, Trivial):
        return field(piece.centre).add_error(piece.radius)
    value = rule_sum(piece, start_precision(piece, field), config)
    return field(value).add_error(piece.error)


def piece_rng(polytope):
    r"""
    Return the random generator of the reconfigurations of the rule on
    ``polytope``, seeded by its vertices.

    EXAMPLES::

        sage: from sieve_integral_brion.evaluate import piece_rng
        sage: box = Polyhedron(vertices=[(1, 1), (2, 1), (1, 3), (2, 3)])
        sage: same = Polyhedron(ieqs=[(-1, 1, 0), (2, -1, 0), (-1, 0, 1), (3, 0, -1)])
        sage: other = Polyhedron(vertices=[(1, 1), (2, 1), (1, 2), (2, 2)])
        sage: piece_rng(box).random() == piece_rng(same).random() != piece_rng(other).random()
        True
    """
    return random.Random(str(sorted(tuple(v) for v in polytope.vertices_list())))


def integrate_piece(piece, field, polytope, border, integrand):
    r"""
    Return a ball of ``field`` containing the integral over ``piece``,
    prepared from ``polytope``, ``border`` and ``integrand`` by
    :func:`~sieve_integral_brion.pieces.prepare_piece`.

    It calls :func:`~sieve_integral_brion.evaluate.evaluate_piece`.
    If the rule has singular nodes, the computation is redone with one
    more node in the last direction; failing that, it's redone with
    a random order of the coordinates; failing that, it's prepared again
    around perturbed centres. The random choices are drawn by
    :func:`piece_rng`. After ``ATTEMPTS`` attempts, it raises
    ``ArithmeticError``, naming the last singular node.

    EXAMPLES:

    On the segment of `t_1 + t_2 = 1` from `(1/4, 3/4)` to `(3/4, 1/4)`, a
    rule with an odd number of nodes has a singular node; with one more,
    the integral is `2 \log 3`::

        sage: import dataclasses
        sage: from sieve_integral_brion.evaluate import counts, integrate_piece
        sage: from sieve_integral_brion.pieces import PLAIN, prepare_piece
        sage: from sieve_integral_brion.rules import RuleConfig
        sage: segment = Polyhedron(vertices=[(1/4, 3/4), (3/4, 1/4)])
        sage: piece = prepare_piece(segment, True, 30)
        sage: m = (piece.degree + 1) // 2
        sage: odd = dataclasses.replace(piece, config=RuleConfig((0, 1), (2 * (m // 2) + 1,)))
        sage: reconfigured = counts["reconfigured"]
        sage: value = integrate_piece(odd, RealBallField(30), segment, True, PLAIN)
        sage: value.overlaps(2 * RealBallField(100)(3).log()), counts["reconfigured"] - reconfigured
        (True, 1)
    """
    rng = piece_rng(polytope)
    config = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return evaluate_piece(piece, field, config)
        except SingularNode as exc:
            singular = exc
        if attempt == ATTEMPTS:
            break
        counts["reconfigured"] += 1
        if attempt in (1, 2):
            config = next_config(singular.config, attempt, bad_pairs(piece.generators), rng)
        else:
            piece = prepare_piece(polytope, border, piece.precision, integrand,
                                  perturbed_centres(polytope, rng))
            config = None
    raise ArithmeticError(f"singular nodes under {ATTEMPTS} configurations of the rule; "
                          f"the last attempt: {singular}")
