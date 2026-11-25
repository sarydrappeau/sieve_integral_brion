r"""The weight `W` of the integrand, and its data on a piece of a polytope.

The integrand is `W(t)/(t_1 \cdots t_d)`, with

.. MATH::

    W(t) = F(t) \prod_f P_f(L_f(t)) \, Q(\psi(t)/t_s),

where:

- `F` is a polynomial over `\QQ`,

- each `P_f = \sum_n c_{f, n} T_n` is a Chebyshev series composed with an
  affine form `L_f`, meaning concretely a :class:`ChebyshevAffineFactor`,

- `Q = \sum_n c_n T_n` is composed with `\psi(t)/t_s`, `\psi` affine;
  concretely, `Q` is a :class:`ChebyshevRatioFactor`.

The arguments of `P_f` and `Q` are assumed to lie in `[-1, 1]` on the
polytope. The coefficients `c_n`, `c_{f, n}` are balls, or rational
numbers. The examples in [DM2026]_ from the Harman sieve (Section 2.5.3)
have `P_f = 1`, `Q = B` the Buchstab function, and `\psi(t) = t_r` (the
ones from Merikoski additionally have a degree-one `F`). The examples from
the density of integers with a localized divisor (Section 3) have `Q = 1`
and two `P_f`'s, representing polynomial approximations to the Buchstab
and Dickman functions.

After rescaling of the coordinates (so that they are centered around
`1`), the weight `W` corresponds to `\Phi(L(t)) Q(\psi(t)/t_s)` of
Section 3.1 of ``mathematics/radial-brion.pdf``, where
(:func:`recentred_weight`):

- `F = \varphi(L_F(t))`, `L_F` the essential forms of `F`
  (:func:`essential_forms`);

- the Chebyshev series (`Q` or `P_f`) are recomposed by an affine-linear
  change of variables, to extend the range to almost all of `[-1, 1]` on
  the polytope piece
  (:func:`~sieve_integral_brion.chebyshev.recentre`);

- `\Phi(y, z) = \varphi(y) \prod_f \tilde P_f(z_f)`, the `\tilde P_f`
  and `\tilde Q` being these rescaled series. The function
  :func:`weight_structure` truncates off the large degree
  coefficients, to a given precision (Section 4.5 ibid.).

The exponents `\beta` of `\Phi` range over `\{|\gamma| \leq \deg \varphi\}
\times \prod_f [0, \deg \tilde P_f]`.

EXAMPLES:

The weight `W(t) = F(t) P(L(t))` with `F(t) = (t_1 + t_2)^2`, `P = T_2 +
T_3/100` and `L(t) = 3 t_1/4 - 7/2`, on the piece `[4, 6]^2`. Its centres
are `(5, 5)`: the rescaled coordinates are `t'_j = t_j/5`, and the
rescaled piece is `[4/5, 6/5]^2`. The piece is full-dimensional, not on
`t_1 + t_2 = 1`, hence ``border=False``::

    sage: from sieve_integral_brion.integrand import *
    sage: t1, t2 = QQ["t1, t2"].gens()
    sage: factor = ChebyshevAffineFactor((0, 0, 1, 1/100), (-7/2, 3/4, 0))   # P(L(t))
    sage: integrand = Integrand((t1 + t2)^2, (factor,))
    sage: vertices = [(4/5, 4/5), (6/5, 4/5), (4/5, 6/5), (6/5, 6/5)]   # of the rescaled piece
    sage: weight = recentred_weight(integrand, centres=(5, 5), vertices=vertices, border=False)

In the rescaled coordinates, `F(t) = \varphi(L_F(t'))` with the essential
form `L_F(t') = 5 t'_1 + 5 t'_2` (``weight.forms``, each form `b + a_1
t'_1 + a_2 t'_2` written `(b, a_1, a_2)`) and `\varphi(y) = y^2`
(``weight.polynomial``). The argument of `P` is `L(t) = 15 t'_1/4 -
7/2`, which ranges over `[-1/2, 1]` on the piece: with `\mu = 1/4` and
`\lambda = 3/4`, the recentring replaces it by its image `5 t'_1 - 5` in
`[-1, 1]`, and `P` by `\tilde P(z) = P(1/4 + 3 z/4)`
(``weight.factors[0]``)::

    sage: weight.forms, weight.polynomial, weight.factors[0]
    (((0, 5, 5),), y^2,
     ChebyshevSeries(coefficients=(-199/640, 4773/6400, 1827/3200, 27/6400), radius=0, form=(-5, 5, 0)))

The bound on `|W|` on the piece is that on `|F|` times that on `|\tilde
P|`. As `L_F` ranges over `[8, 12]`, the former is the sum `12^2` of the
absolute values of the coefficients of `\varphi(10 + 2y) = 100 + 40 y + 4
y^2`; the latter is the sum `2611/1600` of the absolute values of the
coefficients of `\tilde P`::

    sage: weight.mass == 12^2 * 2611/1600
    True

The data of :class:`~sieve_integral_brion._brion_sum.BrionSum` follow::

    sage: structure = weight_structure(weight, allowance=1/100, vertices=vertices,
    ....:                              generators=[(1, 0), (0, 1)])

The rays of the vertex cones of the piece are `\pm e_1, \pm e_2`
(``generators``, up to sign)::

    sage: structure.exponents
    ((0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0), (1, 2), (2, 1), (2, 2))

With ``allowance=1/100``, the last coefficients of `\tilde P` whose
absolute values sum to at most `1/100` times `2611/1600` are left out:
here that of `T_3`, `27/6400`, and ``loss`` `= 12^2 \cdot 27/6400`
bounds the change of `|W|`. The kept part of `\tilde P` is `p(z) =
1827 z^2/1600 + 4773 z/6400 - 1411/1600`, so that `\Phi(y, z) = y^2
p(z)`, and the exponents `\beta = (\beta_y, \beta_z)` range over `[0,
2]^2`, by total degree. The slopes are the values at `e_1` and `e_2`
of the linear parts `5 t'_1 + 5 t'_2` and `5 t'_1` of the forms of `y`
and `z`::

    sage: structure.slopes, structure.loss
    (((5, 5), (5, 0)), 243/400)

Last come the derivatives `\partial^\beta \Phi` at the first vertex
`(4/5, 4/5)`, where `y = 8` and `z = -1`, from `\Phi(8, -1) = 64 p(-1)
= -3109/100` to `\partial_y^2 \partial_z^2 \Phi = 1827/400`::

    sage: structure.derivatives[0]
    (-3109/100, -9843/100, -3109/400, 3654/25, -9843/400, -3109/3200,
     1827/50, -9843/3200, 1827/400)

REFERENCES:

.. [Car2006] E. Carlini, *Reducing the number of variables of a
   polynomial*, in *Algebraic Geometry and Geometric Modeling*, M. Elkadi,
   B. Mourrain and R. Piene (eds.), Mathematics and Visualization,
   Springer, 2006, pp. 237--247.

.. [DM2026] S. Drappeau and A. Mounier, *Computing sieve integrals using
   LattE, and the density of integers with a localized divisor*, preprint,
   2026, :arxiv:`2606.30428`.

AUTHORS:

- Sary Drappeau

AI DISCLOSURE:

Claude Opus 5.5 wrote this module, from a prototype it had written for the
``sieve_integral`` library.

"""
import dataclasses
import itertools
from dataclasses import dataclass

from sage.functions.other import binomial
from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.rational_field import QQ

from .chebyshev import chebyshev_affine_composition, kept_count, recentre, to_monomial


@dataclass(frozen=True)
class ChebyshevAffineFactor:
    r"""
    The factor `\sum_n c_n T_n(L(t))`.

    - ``coefficients`` -- the `c_n`, balls or rational numbers.

    - ``form`` -- `(b, a_1, \ldots, a_d)`: `L(t) = b + \sum_j a_j t_j`.
    """
    coefficients: tuple
    form: tuple


@dataclass(frozen=True)
class ChebyshevRatioFactor:
    r"""
    The factor `\sum_n c_n T_n(\psi(t)/t_s)`.

    - ``coefficients`` -- the `c_n`, balls or rational numbers.

    - ``form`` -- `(b, a_1, \ldots, a_d)`: `\psi(t) = b + \sum_j a_j t_j`.

    - ``divisor`` -- `s - 1`, the position of `t_s` among the coordinates.
    """
    coefficients: tuple
    form: tuple
    divisor: int


@dataclass(frozen=True)
class Integrand:
    r"""
    The weight `W`.

    - ``polynomial`` -- `F`, a polynomial over ``QQ`` in `d` variables, or
      ``None`` for `1`.

    - ``factors`` -- the :class:`ChebyshevAffineFactor` of `W`.

    - ``ratio`` -- its :class:`ChebyshevRatioFactor`, or ``None``.

    - ``nonnegative`` -- whether `W \geq 0` on the polytope.

    - ``essential`` -- ``None``, or the triple ``(border, A, phi)`` of
      :func:`with_essential_forms`; ignored by comparisons.
    """
    polynomial: object = None
    factors: tuple = ()
    ratio: object = None
    nonnegative: bool = False
    essential: object = dataclasses.field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class ChebyshevSeries:
    r"""
    A series `\sum_n c_n T_n(y)` on a piece, `y` being an affine form in the
    rescaled coordinates (the argument of a :class:`ChebyshevAffineFactor`), or
    that form divided by `t_s` (a :class:`ChebyshevRatioFactor`).

    - ``coefficients`` -- the `c_n`, rational numbers.

    - ``radius`` -- the sum of the radii of the given coefficients.

    - ``form`` -- the affine form `(b, a_1, \ldots, a_d)`.
    """
    coefficients: tuple
    radius: object
    form: tuple


@dataclass(frozen=True)
class RecentredWeight:
    r"""
    The weight on a piece, the series not yet truncated.

    - ``forms`` -- the essential forms of `F`, in the rescaled coordinates.

    - ``polynomial`` -- `\varphi`, with `F = \varphi(L_F)`.

    - ``bound`` -- a bound on `|F|` on the piece.

    - ``factors`` -- the :class:`ChebyshevSeries` of the Chebyshev factors.

    - ``ratio`` -- the :class:`ChebyshevSeries` of the factor `Q`, or ``None``.

    - ``divisor`` -- `s - 1`.

    - ``nonnegative`` -- whether `W \geq 0` on the piece.
    """
    forms: tuple
    polynomial: object
    bound: object
    factors: tuple
    ratio: object
    divisor: int
    nonnegative: bool

    @property
    def series(self):
        r"""
        The series of the weight: those of its factors, then that of `Q`.
        """
        return self.factors + ((self.ratio,) if self.ratio is not None else ())

    @property
    def mass(self):
        r"""
        The bound on `|W|` on the piece: that on `|F|` times the sums of the
        absolute values and of the radii of the coefficients of the series.
        """
        return self.bound * prod(sum(abs(c) for c in p.coefficients) + p.radius
                                 for p in self.series)


@dataclass(frozen=True)
class WeightStructure:
    r"""
    The weight on a piece, with the series truncated.

    - ``exponents``, ``slopes``, ``derivatives``, ``ratio`` -- the data of
      :class:`~sieve_integral_brion._brion_sum.BrionSum`.

    - ``bound`` -- a bound on `|\Phi(L(t))|` on the piece.

    - ``weights`` -- the `|q_n| \binom{d + n - 1}{n} \Psi^n` of Theorem 3.4
      of ``mathematics/radial-brion.pdf``, `Q = \sum_n q_n y^n`.

    - ``loss`` -- the bound on the change of `|W|` on the piece by the
      truncations and the radii of the series.
    """
    exponents: tuple
    slopes: object
    derivatives: tuple
    ratio: object
    bound: object
    weights: tuple
    loss: object


def evaluate_affine_form(form, point):
    r"""
    Return the value `b + \sum_j a_j x_j` of ``form`` `(b, a_1, \ldots, a_d)`
    at ``point`` `x`.
    """
    return form[0] + sum(a * x for a, x in zip(form[1:], point))


def essential_forms(polynomial, border):
    r"""
    Return linear forms `A_1, \ldots, A_r` and a polynomial `\varphi` with
    `F(t) = \varphi(\langle A_1, t\rangle, \ldots, \langle A_r, t\rangle)`,
    `F` being ``polynomial``, and `r` the least possible.

    The `A_i` span the orthogonal of the directions `u` in which `F` is
    constant, `\partial_u F = 0`: they are the rows of the reduced echelon
    form of the matrix of the coefficients of the `\partial_j F`. This
    orthogonal is the space of essential variables of `F` ([Car2006]_,
    Proposition 10, whose proof does not use the homogeneity of `F`). Then
    `\varphi(y) = F(t)`, where `t_{p_i} = y_i` for the pivot `p_i` of `A_i`,
    the other `t_j` being `0`. With ``border``, `t_d = 1 - \sum_{j < d} t_j`
    is substituted in `F` first.

    OUTPUT: a pair: the `A_i`, tuples of `d` rational numbers, and
    `\varphi`, a polynomial over ``QQ`` in `r` variables, or a rational
    number if `r = 0`.

    EXAMPLES::

        sage: from sieve_integral_brion.integrand import essential_forms
        sage: t1, t2, t3 = QQ["t1, t2, t3"].gens()
        sage: essential_forms((t1 + t2)^2, False)
        (((1, 1, 0),), y^2)

    Examples 11 and 12 in [Car2006]_::

        sage: essential_forms(t1^3 + t1^2*t2 - 2*t1^2*t3 - 2*t1*t2*t3 + t1*t3^2 + t2*t3^2, False)
        (((1, 0, -1), (0, 1, 1)), y0^3 + y0^2*y1)

    A non-homogeneous example::

        sage: essential_forms(1 + t1 + t2 + t1^2 + 2*t1*t2 + t2^2, False)
        (((1, 1, 0),), y^2 + y + 1)

    On `t_1 + t_2 + t_3 = 1`, `t_3^2 = (1 - t_1 - t_2)^2`, and `t_1 + t_2 +
    t_3` is the constant `1`::

        sage: essential_forms(t3^2, True)
        (((1, 1, 0),), y^2 - 2*y + 1)
        sage: essential_forms(t1 + t2 + t3, True)
        ((), 1)
    """
    t = polynomial.parent().gens()
    if border:
        polynomial = polynomial.subs({t[-1]: 1 - sum(t[:-1])})
    partials = [polynomial.derivative(x) for x in t]
    monomials = sorted({m for p in partials for m in p.monomials()})
    echelon = matrix(QQ, len(monomials), len(t),
                     [p.monomial_coefficient(m) for m in monomials for p in partials]
                     ).echelon_form()
    pivots = echelon.pivots()
    if not pivots:
        return (), QQ(polynomial.constant_coefficient())
    ring = PolynomialRing(QQ, "y", len(pivots))
    image = [ring.zero()] * len(t)
    for i, p in enumerate(pivots):
        image[p] = ring.gen(i)
    return (tuple(tuple(echelon[i]) for i in range(len(pivots))),
            polynomial(*image))


def with_essential_forms(integrand, border):
    r"""
    Return ``integrand`` with the field ``essential`` set to ``(border, A,
    phi)``, ``(A, phi)`` being the :func:`essential_forms` of its
    polynomial. It does nothing and returns ``integrand`` if there is no
    integrand (weight 1), or if this has already been done.

    EXAMPLES::

        sage: from sieve_integral_brion.integrand import Integrand, with_essential_forms
        sage: t1, t2 = QQ["t1, t2"].gens()
        sage: integrand = with_essential_forms(Integrand((t1 + t2)^2), False)
        sage: integrand.essential
        (False, ((1, 1),), y^2)
        sage: with_essential_forms(integrand, False) is integrand
        True
        sage: with_essential_forms(integrand, True).essential
        (True, (), 1)
    """
    if integrand.polynomial is None or (integrand.essential is not None
                                        and integrand.essential[0] == border):
        return integrand
    return dataclasses.replace(integrand, essential=(border,) + essential_forms(
        integrand.polynomial, border))


def midpoints_and_radius(coefficients):
    r"""
    Return the midpoints of ``coefficients``, balls or rational numbers, as
    rational numbers, and the sum of their radii.
    """
    mids = [c.mid().exact_rational() if hasattr(c, "mid") else QQ(c) for c in coefficients]
    radii = [c.rad().exact_rational() for c in coefficients if hasattr(c, "rad")]
    return mids, sum(radii, QQ(0))


def recentred_series(coefficients, values):
    r"""
    Return a tuple with four components:

    - `\mu` and `\lambda` such that `[\mu - \lambda, \mu + \lambda]`
      contains ``values``, the values of the argument `y` of the series of
      ``coefficients`` at the vertices of a piece;

    - the coefficients of the series in the variable `(y - \mu)/\lambda`
      (keeping the midpoints only);

    - and the sum of the radii of ``coefficients``.
    """
    low, high = min(values), max(values)
    if low < -1 or high > 1:
        raise ValueError("the argument of a Chebyshev series leaves [-1, 1] "
                         "on the polytope")
    mu, lam = recentre(low, high)
    mids, radius = midpoints_and_radius(coefficients)
    return mu, lam, tuple(chebyshev_affine_composition(mids, mu, lam)), radius


def recentred_weight(integrand, centres, vertices, border):
    r"""
    Return the :class:`RecentredWeight` of ``integrand`` on the piece
    of vertices ``vertices``, rescaled by ``centres``. See
    :mod:`~sieve_integral_brion.integrand`.

    EXAMPLES:

    The weight `B(t_2/t_1)` on `P = \{t_1 + t_2 = 1, 1/4 \leq t_1 \leq 2/7\}`, on which
    we have `5/2 \leq t_2 / t_1 \leq 3`, with `B` given in by `B(r) =
    T_0(x) + 1/2 T_1(x)`, `x = 2 r - 5 \in [-1, 1]`. Note that the
    role of `B` will be played by the approximants of the Buchstab and
    Dickman functions, whose native rescaling need not correspond to the
    actual range of its argument on `P`: in this example, we have the more precise
    `x \in [0, 1]`::

        sage: from sieve_integral_brion.integrand import *
        sage: ratio = ChebyshevRatioFactor(coefficients=(1, 1/2),  # T0 + 1/2 T1
        ....:                              form=(0, -5, 2),  # -5 t1 + 2 t2
        ....:                              divisor=0)     # divided by t1

    The piece would get recentered at `(15/56, 41/56)`, and in the piece-rescaled
    coordinates `t'_1 = 56 t_1/15`, `t'_2 = 56 t_2/41`, `P` becomes the segment
    `(14/15, 42/41) - (16/15, 40/41)`, and `t_2/t_1 = 41 t'_2/(15 t'_1)`.
    As `x` ranges over `[0, 1]`, the recentring (`\mu = \lambda = 1/2`)
    replaces `x` by `2 x - 1 \in [-1, 1]`, and `B = 1 + x/2` by
    `5/4 + (2 x - 1)/4` (``coefficients=(5/4, 1/4)``), and the form is
    the numerator of `2 x - 1 = 4 t_2/t_1 - 11 = (-11 t'_1 + 164 t'_2/15)/t'_1`::

        sage: centres = (15/56, 41/56)
        sage: segment = [(14/15, 42/41), (16/15, 40/41)]
        sage: weight = recentred_weight(Integrand(ratio=ratio), centres, segment, True)
        sage: weight.ratio
        ChebyshevSeries(coefficients=(5/4, 1/4), radius=0, form=(0, -11, 164/15))
    """
    d = len(centres)
    if integrand.polynomial is None:
        forms, polynomial, bound = (), QQ(1), QQ(1)
    else:
        if integrand.polynomial.parent().ngens() != d:
            raise ValueError(f"the polynomial must have {d} variables")
        _, linear, polynomial = with_essential_forms(integrand, border).essential
        forms = tuple((QQ(0),) + tuple(a * c for a, c in zip(A, centres)) for A in linear)
        if forms:
            # the sum of the absolute values of the coefficients of phi,
            # recentred on the box of the ranges of the forms
            ranges = [[evaluate_affine_form(f, v) for v in vertices] for f in forms]
            shifted = polynomial(*[(min(r) + max(r)) / 2 + (max(r) - min(r)) / 2 * y
                                   for r, y in zip(ranges, polynomial.parent().gens())])
            bound = sum(abs(c) for c in shifted.coefficients())
        else:
            bound = abs(polynomial)
    factors = []
    for factor in integrand.factors:
        if len(factor.form) != d + 1:
            raise ValueError(f"the form of a factor must have {d + 1} coefficients")
        b, *A = factor.form
        form = (QQ(b),) + tuple(QQ(a) * c for a, c in zip(A, centres))
        mu, lam, coefficients, radius = recentred_series(
            factor.coefficients, [evaluate_affine_form(form, v) for v in vertices])
        factors.append(ChebyshevSeries(coefficients, radius, ((form[0] - mu) / lam,)
                                       + tuple(a / lam for a in form[1:])))
    ratio, divisor = None, None
    if integrand.ratio is not None:
        if len(integrand.ratio.form) != d + 1:
            raise ValueError(f"the form of the ratio must have {d + 1} coefficients")
        divisor = integrand.ratio.divisor
        b, *A = integrand.ratio.form
        # psi(c t)/c_s, whose quotient by t_s is psi/t_s in the original
        # coordinates; then (psi(c t)/c_s - mu t_s)/lam
        form = [QQ(b) / centres[divisor]] + [QQ(a) * c / centres[divisor]
                                             for a, c in zip(A, centres)]
        mu, lam, coefficients, radius = recentred_series(
            integrand.ratio.coefficients,
            [evaluate_affine_form(form, v) / v[divisor] for v in vertices])
        form[1 + divisor] -= mu
        ratio = ChebyshevSeries(coefficients, radius, tuple(a / lam for a in form))
    return RecentredWeight(forms, polynomial, bound, tuple(factors), ratio, divisor,
                           integrand.nonnegative)


def derivative_table(polynomial, exponents, points):
    r"""
    Return the derivatives `\partial^\beta` of ``polynomial`` at each point
    of ``points``, and order given by the exponents `\beta` of ``exponents``.
    The list ``exponents`` should be closed under taking smaller exponents,
    and ordered as `(\beta_i)_i` in such a way that two exponents
    `\beta_i, \beta_j` satisfying `\beta_i \leq \beta_j` component-wise should have
    `i \leq j` (each exponent vector comes after the ones of lesser order).
    """
    gens = polynomial.parent().gens()
    derivatives = {}
    for beta in exponents:
        if not any(beta):  # (0, ..., 0)
            derivatives[beta] = polynomial
            continue
        # index of the first non-zero coordinate
        f = next(i for i, e in enumerate(beta) if e)
        below = tuple(e - (i == f) for i, e in enumerate(beta))
        # compute the derivative from the one right below it
        derivatives[beta] = derivatives[below].derivative(gens[f])
    return [[derivatives[beta](*point) for beta in exponents] for point in points]


def weight_structure(weight, allowance, vertices, generators):
    r"""
    Return the :class:`WeightStructure` of the :class:`RecentredWeight`
    ``weight`` on a piece of vertices ``vertices``, whose cones have the
    rays ``generators`` up to sign. The `d` of Theorem 3.4 of
    ``mathematics/radial-brion.pdf`` is the number of coordinates of the
    vertices, also on a piece of `\sum_j t_j = 1`.

    Each series `\sum_n c_n T_n` keeps the least number of coefficients
    with which those left out have a sum of absolute values at most
    ``allowance`` times `\sum_n |c_n|` plus the sum of the radii
    (:func:`~sieve_integral_brion.chebyshev.kept_count`).

    EXAMPLES:

    The weight `(t_1 + t_2 + 1) (T_0 + T_1/100)(56 t_1 - 15)`, on the
    segment of :func:`recentred_weight`. There `t_1 + t_2 + 1 = 2`, and
    `56 t_1 - 15` ranges over `[-1, 1]`::

        sage: from sieve_integral_brion.integrand import *
        sage: t1, t2 = QQ["t1, t2"].gens()
        sage: centres = (15/56, 41/56)
        sage: segment = [(14/15, 42/41), (16/15, 40/41)]
        sage: factor = ChebyshevAffineFactor((1, 1/100), (-15, 56, 0))
        sage: weight = recentred_weight(Integrand(t1 + t2 + 1, (factor,)), centres, segment, True)

    With the allowance `1/8`, the term `T_1(56 t_1 - 15)/100` is discarded,
    and ``loss`` `= 2/100` bounds the change of `|W|`; the remaining weight
    is the constant `2`. ``bound`` is its trivial bound (still `2`). For each
    generator `g`, ``slopes`` holds the values `\langle A_f, g\rangle`
    for `A_f` the linear parts of the forms involved in the weight; here the only form
    is `56 t_1 - 15 = 15 t'_1 - 15` in the rescaled coordinates `t'_1 = 56 t_1/15`,
    and the only generator is `g = (41, -15)`, so the slope is `15 \cdot 41 = 615`. For
    each vertex `v`, ``derivatives`` holds the `\partial^\beta \Phi(L(v))`,
    `\beta` running over ``exponents``, here only the constant value `\Phi = 2`::

        sage: weight_structure(weight=weight,
        ....:                  allowance=1/8,
        ....:                  vertices=segment,
        ....:                  generators=[(41, -15)])
        WeightStructure(exponents=((0,),), slopes=((615,),), derivatives=((2,), (2,)),
                        ratio=None, bound=2, weights=(1,), loss=1/50)
    """
    kept, sups, losses = [], [], []
    for series in weight.series:
        full = sum(abs(c) for c in series.coefficients)
        K, tail = kept_count(series.coefficients, allowance * (full + series.radius))
        kept.append(series.coefficients[:K])
        sups.append(sum(abs(c) for c in kept[-1]))
        losses.append(series.radius + tail)
    loss = weight.bound * (prod(s + l for s, l in zip(sups, losses)) - prod(sups))
    # the blocks of Phi: phi of the essential forms, then the factors
    blocks = []
    if weight.forms:
        degree = weight.polynomial.total_degree()
        exponents = sorted((e for e in itertools.product(range(degree + 1),
                                                         repeat=len(weight.forms))
                            if sum(e) <= degree), key=sum)
        blocks.append((weight.forms, weight.polynomial, exponents))
    else:
        # phi constant: its block has no forms and the exponent ().
        # weight.polynomial is then a Rational so we type cast to a polynomial
        blocks.append(((), PolynomialRing(QQ, "y", 0)(weight.polynomial), [()]))
    for series, coefficients in zip(weight.factors, kept):
        polynomial = to_monomial(coefficients)
        blocks.append(((series.form,), polynomial,
                       [(e,) for e in range(max(polynomial.degree(), 0) + 1)]))

    # The computation of d^beta Phi(L(v)) follows (the composition makes them
    # a tad complicated).

    # tables[b][v][i]: the derivative of order exponents_b[i] of the
    # polynomial p_b of the block b, at the values of its forms at the vertex v
    tables = [derivative_table(polynomial, exponents,
                               [[evaluate_affine_form(f, v) for f in forms] for v in vertices])
              for forms, polynomial, exponents in blocks]

    # The derivative orders, sorted in non-decreasing ell^1 norm (see BrionSum)
    pairs = sorted(((sum((b[2][i] for b, i in zip(blocks, c)), ()), c)
                    for c in itertools.product(*[range(len(b[2])) for b in blocks])),
                   key=lambda pair: sum(pair[0]))
    exponents = tuple(beta for beta, c in pairs)

    # derivatives[v]: the d^beta Phi(L(v)), in the order of the exponents
    derivatives = tuple(tuple(prod(t[v][i] for t, i in zip(tables, c)) for beta, c in pairs)
                        for v in range(len(vertices)))

    # the forms of the arguments of Phi, in the order of the blocks; their
    # slopes <A_f, g> are the values of their linear parts at the generators
    forms = [f for b in blocks for f in b[0]]
    slopes = (tuple(tuple(evaluate_affine_form((0,) + f[1:], g) for f in forms)
                    for g in generators)
              if forms else None)

    # the bound on |F| times the product of the sums of |c_n| for the kept P_f
    bound = weight.bound * prod(sups[:len(weight.factors)])
    if weight.ratio is None:
        return WeightStructure(exponents, slopes, derivatives, None, bound, (QQ(1),), loss)

    # the q_n of the kept Q = sum_n q_n y^n, or [0] if it is 0
    q = to_monomial(kept[-1]).list() or [QQ(0)]

    # Q's argument is psi(t)/t_s; psi is affine, so Psi = sup |psi| is
    # reached at a vertex
    psi = weight.ratio.form
    values = [evaluate_affine_form(psi, v) for v in vertices]
    Psi = max(abs(x) for x in values)
    d = len(vertices[0])

    # (q_n, s - 1, the slopes <psi_1, g> of the generators, the psi(v))
    ratio = (tuple(q), weight.divisor,
             tuple(evaluate_affine_form((0,) + psi[1:], g) for g in generators),
             tuple(values))

    # the terms |q_n| binom(d + n - 1, n) Psi^n of Theorem 3.4
    weights = tuple(abs(c) * binomial(d + n - 1, n) * Psi**n for n, c in enumerate(q))

    return WeightStructure(exponents, slopes, derivatives, ratio, bound, weights, loss)
