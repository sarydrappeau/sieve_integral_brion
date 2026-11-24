r"""
Positive cubature rules on the standard simplex.

The standard simplex of `\RR^d` is `\Delta = \{x \in \RR_{\geq 0}^d,
\sum_j x_j = 1\}`, with the measure `dx_1 \cdots dx_{d-1}` (total mass
`1/(d-1)!`). With `k = d - 1`, the map

.. MATH::

    x_i = u_i \prod_{i' < i} (1 - u_{i'}) \quad (1 \leq i \leq k), \qquad
    x_d = \prod_{i \leq k} (1 - u_i)

sends `[0, 1]^k` onto `\Delta`, with Jacobian `\prod_{i \leq k} (1 -
u_i)^{k - i}`. The present module outputs Stroud's conical product rule
[Str1971]_, the tensor product of Gauss quadrature rules having, in the
direction `u_i`, `m_i` points and weight `(1 - u)^{k - i}`. Its weights are
positive, and it integrates exactly the polynomials of degree at
most `\min_i (2 m_i - 1)` on `\Delta`.

Nodes and weights are balls. Each node is certified to contain one root of the
Jacobi polynomial by a change of sign, or by Sage's exact root isolation
(``real_roots``).

The rule may be applied to a simplex with permuted coordinates (that's
one of the ways we make the nodes avoid singularities in the functions
:func:`choose_config` and :func:`next_config`). A :class:`RuleConfig`
records the permutation of coordinates, and the numbers of points
`m_i` in each direction.

EXAMPLES::

    sage: from sieve_integral_brion.rules import conical_rule
    sage: rule = conical_rule([2, 2], 53)
    sage: len(rule), sum(w for _, w in rule)
    (4, [0.500000000000000 +/- ...e-16])

REFERENCES:

.. [DR1967] P. J. Davis and P. Rabinowitz, *Numerical Integration*, Blaisdell,
   1967.

.. [GR2007] I. S. Gradshteyn and I. M. Ryzhik, *Table of Integrals, Series,
   and Products*, 7th edition, Academic Press, 2007.

.. [Str1971] A. H. Stroud, *Approximate Calculation of Multiple Integrals*,
   Prentice-Hall, 1971, Section 2.4.

AUTHORS:

- Sary Drappeau

AI DISCLOSURE:

Claude Opus 5.5 wrote this module, from a prototype it had written for the
``sieve_integral`` library.
"""
import itertools
from dataclasses import dataclass
from functools import lru_cache

from sage.arith.misc import binomial
from sage.rings.rational_field import QQ
from sage.rings.real_arb import RealBallField
from sage.rings.real_mpfi import RealIntervalField
from sage.rings.real_mpfr import RealField
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing


@lru_cache(maxsize=None)
def jacobi_polynomial(m, a):
    r"""
    Return `P_m^{(a, 0)}(2u - 1)`, whose roots are the nodes of the `m`-point
    Gauss rule on `[0, 1]` for the weight `(1 - u)^a`.

    EXAMPLES::

        sage: from sieve_integral_brion.rules import jacobi_polynomial
        sage: jacobi_polynomial(2, 0)
        6*u^2 - 6*u + 1
        sage: jacobi_polynomial(2, 6)
        45*u^2 - 18*u + 1
    """
    u = PolynomialRing(QQ, "u").gen()
    return sum(binomial(m + a, m - s) * binomial(m, s) * (u - 1)**s * u**(m - s)
               for s in range(m + 1))


@lru_cache(maxsize=None)
def jacobi_rational_roots(m, a):
    r"""
    Return the rational roots of :func:`jacobi_polynomial`, in increasing
    order.

    EXAMPLES:

    The middle node of the Gauss-Legendre rules with an odd number of points,
    and the only other case for `2 \leq m \leq 12`, `a \leq 7`::

        sage: from sieve_integral_brion.rules import jacobi_rational_roots
        sage: jacobi_rational_roots(3, 0)
        (1/2,)
        sage: jacobi_rational_roots(2, 6)
        (1/15, 1/3)
        sage: jacobi_rational_roots(4, 3)
        ()
    """
    return tuple(sorted(jacobi_polynomial(m, a).roots(QQ, multiplicities=False)))


@lru_cache(maxsize=None)
def gauss_jacobi(m, a, prec):
    r"""
    Return the nodes and weights of the `m`-point Gauss rule on `[0, 1]` for
    the weight `(1 - u)^a`, as balls of precision ``prec``.

    Each root of `q =` :func:`jacobi_polynomial` is enclosed in an interval
    of width `2^{1 - e}`, `e = ` ``prec`` `+ 4m + 20`, across which `q`
    changes sign, and the weights are `1/(u (1 - u) q'(u)^2)`, computed in
    ball arithmetic at precision `e`. These are the weights of [DR1967]_,
    (2.7.8), rewritten with [GR2007]_, (8.961.2-3), from the variable
    `x \in [-1, 1]` of these references to `u = (1 + x)/2`.

    OUTPUT: a pair of tuples of ``m`` balls, the nodes in increasing order.

    EXAMPLES:

    The two-point Gauss-Legendre rule, nodes `1/2 \mp 1/(2 \sqrt 3)`::

        sage: from sieve_integral_brion.rules import gauss_jacobi
        sage: nodes, weights = gauss_jacobi(2, 0, 53)
        sage: nodes
        ([0.2113248654051871 +/- ...], [0.7886751345948129 +/- ...])
        sage: weights
        ([0.5000000000000000 +/- ...], [0.5000000000000000 +/- ...])

    The weights add up to `\int_0^1 (1 - u)^a du`::

        sage: sum(gauss_jacobi(7, 3, 100)[1]).contains_exact(1/4)
        True
    """
    q = jacobi_polynomial(m, a)
    extra = prec + 4 * m + 20
    field = RealBallField(extra)
    delta = QQ(2)**(-extra)
    guesses = sorted(x.exact_rational() for x in
                     q.change_ring(RealField(extra + 20)).roots(multiplicities=False))
    ends = [(x - delta, x + delta) for x in guesses]
    if (len(ends) == m and 0 < ends[0][0] and ends[-1][1] < 1
            and all(b < c for (_, b), (c, _) in zip(ends, ends[1:]))
            and all(q(lo).sign() * q(hi).sign() < 0 for lo, hi in ends)):
        nodes = [field(x, delta) for x in guesses]
    else:    # slower alternative (have Sage find the roots in RIF)
        roots = q.roots(RealIntervalField(extra + 20), multiplicities=False)
        if len(roots) != m:
            raise ArithmeticError(f"found {len(roots)} roots of the Jacobi "
                                  f"polynomial of degree {m}")
        nodes = sorted((field(root) for root in roots), key=lambda x: x.mid())
    dq = q.derivative()
    weights = [1 / (x * (1 - x) * dq(x)**2) for x in nodes]
    ball = RealBallField(prec)
    return tuple(ball(x) for x in nodes), tuple(ball(w) for w in weights)


def conical_factors(ms, prec):
    r"""
    Return the one-dimensional rules of the conical product with ``ms[i]``
    points in the direction `u_{i+1}`, on the simplex of `\RR^{k+1}`, `k =`
    ``len(ms)``: the pairs ``(nodes, weights)`` of :func:`gauss_jacobi`
    for the weights `(1 - u)^{k - 1 - i}`.

    EXAMPLES::

        sage: from sieve_integral_brion.rules import conical_factors, gauss_jacobi
        sage: factors = conical_factors([3, 2], 53)
        sage: factors[0] is gauss_jacobi(3, 1, 53), factors[1] is gauss_jacobi(2, 0, 53)
        (True, True)
    """
    k = len(ms)
    return [gauss_jacobi(m, k - 1 - i, prec) for i, m in enumerate(ms)]


def conical_rule(ms, prec, order=None):
    r"""
    Return the nodes and weights of the conical product rule on the standard
    simplex of `\RR^{k+1}`, `k =` ``len(ms)``, with ``ms[i]`` points in the
    direction `u_{i+1}`.

    INPUT:

    - ``ms`` -- a list of positive integers.

    - ``prec`` -- the precision of the balls.

    - ``order`` (default: the identity) -- a permutation of ``range(k + 1)``:
      the coordinate ``order[i]`` of each node is the `x_{i+1}` of the map
      above.

    OUTPUT: a list of pairs ``(node, weight)``, each node a tuple of `k + 1`
    balls.

    EXAMPLES:

    On the simplex of `\RR^3`, `\int_\Delta x_1 x_3^2 = 1!\,2!/5! = 1/60`::

        sage: from sieve_integral_brion.rules import conical_rule
        sage: rule = conical_rule([2, 2], 100)
        sage: sum(w * x[0] * x[2]^2 for x, w in rule).contains_exact(1/60)
        True

    With the coordinates in another order, the rule is another positive rule
    of the same degree::

        sage: other = conical_rule([2, 2], 100, order=[2, 0, 1])
        sage: all(a.identical(rule[0][0][i]) for a, i in zip(other[0][0], (1, 2, 0)))
        True
        sage: sum(w * x[0] * x[2]^2 for x, w in other).contains_exact(1/60)
        True
    """
    ball = RealBallField(prec)
    factors = conical_factors(ms, prec)
    k = len(ms)
    if order is None:
        order = range(k + 1)
    rule = []
    for index in itertools.product(*(range(m) for m in ms)):
        prefix, weight, x = ball.one(), ball.one(), []
        for (nodes, weights), j in zip(factors, index):
            x.append(prefix * nodes[j])
            prefix *= 1 - nodes[j]
            weight *= weights[j]
        x.append(prefix)
        node = [None] * (k + 1)
        for i, coordinate in enumerate(order):
            node[coordinate] = x[i]
        rule.append((tuple(node), weight))
    return rule


@dataclass(frozen=True)
class RuleConfig:
    r"""
    A configuration of the conical product rule on the simplex of `\RR^d`:
    the permutation ``order`` of ``range(d)`` and the `d - 1` numbers of
    nodes ``ms``, as in :func:`conical_rule`.
    """
    order: tuple
    ms: tuple


def choose_config(m, d, bad=()):
    r"""
    Return a :class:`RuleConfig` on the simplex of `\RR^d` with at least
    ``m`` nodes in each direction, whose nodes avoid the hyperplanes `x_i =
    x_j`, `\{i, j\}` in ``bad``.

    In the direction `u_i`, `i < d - 1`, the number of nodes is the least
    `m_i \geq m` whose Jacobi polynomial has no rational root. In the last
    direction, an odd number of nodes produces the node `u_{d-1} = 1/2`, at
    which the two coordinates that this direction fills (`x_{d-1}` and `x_d`
    of the map above, that is the coordinates ``order[d - 2]`` and
    ``order[d - 1]`` of the node) are equal. In this situation we first try
    to permute order to put in those positions a pair `\{i, j\}` not in
    ``bad``, the identity's own pair `\{d - 2, d - 1\}` if possible.
    If every pair is in ``bad``, the last direction gets `m + 1` nodes
    instead (this however tends to raise the computation time, which is why
    we do it as second line only).

    INPUT:

    - ``m`` -- a positive integer.

    - ``d`` -- an integer at least `2`.

    - ``bad`` (default: none) -- a collection of frozen sets of two
      coordinates (:func:`~sieve_integral_brion.geometry.bad_pairs`).

    EXAMPLES::

        sage: from sieve_integral_brion.rules import choose_config
        sage: choose_config(3, 3)
        RuleConfig(order=(0, 1, 2), ms=(3, 3))
        sage: choose_config(3, 3, {frozenset({1, 2})})
        RuleConfig(order=(1, 0, 2), ms=(3, 3))
        sage: choose_config(3, 2, {frozenset({0, 1})})
        RuleConfig(order=(0, 1), ms=(4,))

    The Jacobi polynomial of `m = 2` points for the weight `(1 - u)^6` has
    the roots `1/15` and `1/3`::

        sage: choose_config(2, 9).ms
        (2, 3, 2, 2, 2, 2, 2, 2)
    """
    k = d - 1
    ms = []
    for i in range(k - 1):
        n = m
        while jacobi_rational_roots(n, k - 1 - i):
            n += 1
        ms.append(n)
    order = tuple(range(d))
    last = m
    if m % 2:
        good = next(((i, j) for j in reversed(range(d)) for i in reversed(range(j))
                     if frozenset((i, j)) not in bad), None)
        if good is None:
            last = m + 1
        else:
            order = tuple(c for c in order if c not in good) + good
    ms.append(last)
    return RuleConfig(order, tuple(ms))


def next_config(config, attempt, bad, rng):
    r"""
    Return the configuration of the rule for the attempt ``attempt``,
    following ``config``: for the attempt `1`, one more node in the last
    direction; for the attempt `2`, a random order. If the last direction
    then has an odd number of nodes, and the last two coordinates are in
    ``bad``, it gets one more.

    INPUT:

    - ``config`` -- a :class:`RuleConfig`.

    - ``attempt`` -- `1` or `2`.

    - ``bad`` -- as in :func:`choose_config`.

    - ``rng`` -- a ``random.Random`` instance, for the order.

    EXAMPLES::

        sage: import random
        sage: from sieve_integral_brion.rules import RuleConfig, next_config
        sage: bad = {frozenset({1, 2})}
        sage: next_config(RuleConfig((1, 0, 2), (3, 3)), 1, bad, random.Random(int(1)))
        RuleConfig(order=(1, 0, 2), ms=(3, 4))
        sage: next_config(RuleConfig((0, 1, 2), (3, 4)), 1, bad, random.Random(int(1)))
        RuleConfig(order=(0, 1, 2), ms=(3, 6))
        sage: sorted(next_config(RuleConfig((0, 1, 2), (3, 4)), 2, bad, random.Random(int(1))).order)
        [0, 1, 2]
    """
    order, ms = list(config.order), list(config.ms)
    if attempt == 1:
        ms[-1] += 1
    else:
        rng.shuffle(order)
    if ms[-1] % 2 and frozenset(order[-2:]) in bad:
        ms[-1] += 1
    return RuleConfig(tuple(order), tuple(ms))
