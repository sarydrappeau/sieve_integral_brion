r"""
Integrals with a weight over polytopes of dimension `k \leq 1`, in closed form.

On the interval `[a, b]` of `\RR_{>0}` (`d = 1`), and on the segment of
`t_1 + t_2 = 1` on which `t_1` runs over `[a, b]` (`d = 2`), the weight
`W(t)` of :mod:`~sieve_integral_brion.integrand`, with the midpoints of
its coefficients, is a rational function of `\tau = t_1`: `F` and the
Chebyshev factors are polynomials in `\tau`, and `Q(\psi(t)/t_s)` is a
polynomial in `1/t_s`. So is the integrand `W(t)/(t_1 \cdots t_d)`, with
possible poles at `0` and `1` only. Its integral is computed explicitly
by Maxima as a rational number plus rational multiples of `\log(b/a)` and
of `\log((1 - b)/(1 - a))`. When `k = 0`, i.e. on the point `t_1 = 1`
of `\RR`, the integral is `W(1)`.

The radii of the coefficients change `|W|` by at most `\sup |F|
(\prod_f (S_f + r_f) - \prod_f S_f)`, `S_f` and `r_f` being the sums of
the absolute values of the midpoints and of the radii of each series. This
times the integral of `1/(t_1 \cdots t_d)` account for the induced error.

EXAMPLES:

`\int_1^3 (1 + t) T_2(t - 2) \, dt/t = -26/3 + 7 \log 3`::

    sage: from sieve_integral_brion.segments import closed_form_integral
    sage: from sieve_integral_brion.integrand import ChebyshevAffineFactor, Integrand
    sage: t = PolynomialRing(QQ, "t", 1).gen()
    sage: integrand = Integrand(1 + t, (ChebyshevAffineFactor((0, 0, 1), (-2, 1)),))
    sage: R = RealBallField(50)
    sage: value = closed_form_integral(Polyhedron([(1,), (3,)]), False, integrand, 50, R)
    sage: value.overlaps(-26/3 + 7 * R(3).log()), value.rad() < 2^-50
    (True, True)

On the point `t_1 = 1`, `W(1) = 2 T_2(-1) = 2`::

    sage: closed_form_integral(Polyhedron([(1,)]), True, integrand, 50, R)
    2.00000000000000

AUTHORS:

- Sary Drappeau

AI DISCLOSURE:

Claude Opus 5.5 wrote this module.
"""
from sage.misc.misc_c import prod
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.rational_field import QQ
from sage.rings.real_arb import RealBallField
from sage.symbolic.ring import SR

from .chebyshev import to_monomial
from .integrand import evaluate_affine_form, recentred_weight


def closed_form_value(function, a, b, precision):
    r"""
    Return a ball containing `\int_a^b f`, `f` being ``function``, a
    rational function over ``QQ`` without poles in `[a, b]`, with a radius
    at most `2^{-p - 32}`, `p` being ``precision``.

    Maxima gives the integral as a rational number plus rational multiples
    of logarithms of rational numbers. These terms can be large and cancel:
    Arb evaluates their sum with more bits until the radius is small enough.

    EXAMPLES:

    `\int_{1/4}^{1/2} d\tau/(\tau (1 - \tau)) = \log 3`::

        sage: from sieve_integral_brion.segments import closed_form_value
        sage: tau = QQ["tau"].gen()
        sage: value = closed_form_value(1 / (tau * (1 - tau)), 1/4, 1/2, 50)
        sage: value.overlaps(RealBallField(200)(3).log()), value.rad() < 2^-82
        (True, True)
    """
    x = SR.var("x")
    integral = function(x).integral(x, a, b, algorithm="maxima")
    bits = precision + 64
    while True:
        value = RealBallField(bits)(integral)
        if value.rad() <= 2**(-precision - 32):
            return value
        bits *= 2


def closed_form_integral(polytope, border, integrand, precision, field):
    r"""
    Return a ball of ``field`` containing the integral of ``integrand``
    over ``polytope``, of dimension `k = d - ` ``border`` `\leq 1`, at the
    precision ``precision``.
    """
    d = polytope.ambient_dimension()
    if d == 0:
        # as sieve_integral, on the point of RR^0
        return field.zero()
    vertices = polytope.vertices_list()
    weight = recentred_weight(integrand, (QQ(1),) * d, vertices, border)
    tau = PolynomialRing(QQ, "tau").gen()
    point = (tau,) if d == 1 else (tau, 1 - tau)
    function = tau.parent().fraction_field()(
        weight.polynomial(*[evaluate_affine_form(f, point) for f in weight.forms])
        if weight.forms else weight.polynomial)
    for series in weight.factors:
        function *= to_monomial(series.coefficients)(evaluate_affine_form(series.form, point))
    if weight.ratio is not None:
        function *= to_monomial(weight.ratio.coefficients)(
            evaluate_affine_form(weight.ratio.form, point) / point[weight.divisor])
    sums = [sum(abs(c) for c in s.coefficients) for s in weight.series]
    loss = weight.bound * (prod(S + s.radius for S, s in zip(sums, weight.series))
                           - prod(sums))
    if d == 1 and border:
        value = RealBallField(precision + 64)(function(1))
        return field(value.add_error(loss))
    a, b = min(v[0] for v in vertices), max(v[0] for v in vertices)
    value = closed_form_value(function / prod(point), a, b, precision)
    mass = closed_form_value(1 / prod(point), a, b, precision)
    return field(value.add_error(loss * mass))
