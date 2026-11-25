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
import dataclasses

from sage.geometry.polyhedron.constructor import Polyhedron
from sage.rings.rational_field import QQ
from sage.rings.real_arb import RealBallField

from .dde import BuchstabB
from .evaluate import counts, integrate_piece
from .inputs import (as_polytope, check_positive_orthant, eqn_sum_one, is_polytope_on_sum1,
                     parse_polynomial, variable_indices)
from .integrand import (ChebyshevAffineFactor, ChebyshevRatioFactor, Integrand,
                        with_essential_forms)
from .pieces import PLAIN, prepare_piece
from .segments import closed_form_integral

# B is computed with these bits above the precision, to mitigate the error loss
# coming from its coefficients
BUCHSTAB_EXTRA_BITS = 10


def integrate_many(tasks, processes=None, field=None, verbose=0):
    r"""
    Return a ball containing the sum of the integrals of the tasks ``tasks``.

    INPUT:

    - ``tasks`` -- tuples ``(ieqs, eqns, integrand, options)``: a polytope
      by its inequalities and equations (either no equations, or that of
      `\sum_j t_j = 1`), in the open positive orthant, an
      :class:`~sieve_integral_brion.integrand.Integrand`, and a dictionary
      with the key ``"precision"``, the integer `p`: each polytope is taken
      as one piece, whose error is at most `2^{-p}`. The polytopes of
      dimension at most `1` are integrated in closed form, and those of
      lower dimension than their space are left out.

    - ``processes`` (default: ``None``) -- to be implemented in the next
      commit (the computation is not parallelized in this one yet).

    - ``field`` (default: ``None``) -- the ball field of the result, by
      default that of the largest precision of the tasks.

    - ``verbose`` (default: 0) -- if positive, print the numbers of nodes,
      of nodes computed again at a higher precision, of singular nodes and
      of reconfigurations of the rule.

    EXAMPLES:

    The integral of `1 + 2 t_1` over `P = \{1/10 < t_1 < t_2 < t_3, \sum_j
    t_j = 1\}`, compared with its value as one-dimensional integral

    .. MATH::

        \int_P \frac{(1 + 2 t_1) dt_1 dt_2}{t_1 t_2 t_3} = \int_{1/10}^{1/3}
        \frac{1 + 2 t}{t (1 - t)} \log\frac{1 - 2 t}{t} dt,

    computed by Arb::

        sage: from sieve_integral_brion.api import integrate_many
        sage: from sieve_integral_brion.integrand import Integrand
        sage: t1 = QQ["t1, t2, t3"].gen(0)
        sage: P = [(-1/10, 1, 0, 0), (0, -1, 1, 0), (0, 0, -1, 1)]
        sage: value = integrate_many([(P, [(-1, 1, 1, 1)], Integrand(1 + 2 * t1),
        ....:                          {"precision": 20})])
        sage: C = ComplexBallField(100)
        sage: expected = C.integral(lambda t, a: (1 + 2 * t) * ((1 - 2 * t) / t).log(analytic=a)
        ....:                       / (t * (1 - t)), 1/10, 1/3).real()
        sage: value.overlaps(expected), value.rad() < 2^-20
        (True, True)
    """
    if field is None:
        field = RealBallField(max((options["precision"] for *_, options in tasks), default=20))
    values = []
    before = dict(counts)
    for ieqs, eqns, integrand, options in tasks:
        polytope = Polyhedron(ieqs=ieqs, eqns=eqns or None, base_ring=QQ)
        border = bool(eqns)
        k = polytope.ambient_dimension() - border
        if polytope.dimension() < k:
            continue
        precision = options["precision"]
        if k <= 1:
            values.append(closed_form_integral(polytope, border, integrand, precision, field))
            continue
        integrand = with_essential_forms(integrand, border)
        piece = prepare_piece(polytope, border, precision, integrand)
        values.append(integrate_piece(piece, field, polytope, border, integrand))
    if verbose > 0:
        change = {name: counts[name] - before[name] for name in counts}
        print(f"integrate_many: {change['nodes']} nodes, {change['retried']} computed "
              f"again, {change['singular']} singular, {change['reconfigured']} "
              f"reconfigurations")
    # added smallest first, with 64 more bits than the field, for the
    # rounding of the sum
    wide = RealBallField(field.precision() + 64)
    values.sort(key=lambda v: (abs(v.mid()), v.mid(), v.rad()))
    return field(sum((wide(v) for v in values), wide.zero()))


def polytope_task(polytope, border, integrand, precision):
    r"""
    Return the task of :func:`integrate_many` for ``integrand`` on ``polytope``.
    """
    eqns = [eqn_sum_one(polytope.ambient_dimension())] if border else []
    return (polytope.inequalities_list(), eqns, integrand, {"precision": precision})


def weighted_polytope(polytope_data, facteur):
    r"""
    Return the polytope of ``polytope_data``, its variables, and whether it
    lies on `\sum_j t_j = 1`, ``None`` if it is empty, after checking it
    and ``facteur`` for the functions with a weight.
    """
    if facteur is not None:
        # Cutting the polytope into pieces is planned for later.
        raise NotImplementedError("only facteur=None is supported")
    polytope, variables = as_polytope(polytope_data)
    if polytope.is_empty():
        return polytope, variables, None
    check_positive_orthant(polytope)
    return polytope, variables, is_polytope_on_sum1(polytope)


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

    - ``verbose`` (default: 0) -- if positive, print the counts of
      :func:`integrate_many`, then the value.

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
        sage: value.overlaps(C.integral(lambda x, a: (6 - x).log(analytic=a) / x,
        ....:                           1, 5).real() / 2)
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
    value = integrate_many([polytope_task(polytope, border, PLAIN, precision)],
                           processes, field, verbose)
    if verbose > 0:
        print(f"sieve_integral returning {value}")
    return value


def sieve_integral_harman(polytope_data, buchstab_variables, integrand_polynomial_data=None,
                          precision=20, facteur=None, processes=None, verbose=0,
                          engine=None):
    r"""
    Return a RealBall containing `\int_P B(t_r/t_s) F(t) \, dt_1 \cdots
    dt_k/(t_1 \cdots t_d)`, where `B(u) = u \omega(u)`, `\omega` being
    Buchstab's function, and `F` is a polynomial.

    The polytope `P \subset \RR_{>0}^d` is either full-dimensional (`k =
    d`), or on the hyperplane `\sum_j t_j = 1` (`k = d - 1`). It is cut by
    the intervals `[n, n + 1]` of `t_r/t_s` on which `B` is a Chebyshev
    series (computed ``BUCHSTAB_EXTRA_BITS`` bits above the precision), and
    the parts are integrated together by :func:`integrate_many`. If `k \leq
    1`, the integral is in closed form.

    INPUT:

    - ``polytope_data`` -- a polytope, or an iterable of symbolic linear
      equalities and inequalities.

    - ``buchstab_variables`` -- the pair `(t_r, t_s)`, of symbolic variables
      or of the indices `(r - 1, s - 1)`; with ``None``, there is no factor
      `B`.

    - ``integrand_polynomial_data`` (default: ``None``, meaning `F = 1`) --
      `F`, a symbolic expression in the variables of ``polytope_data``, or a
      polynomial in `d` variables.

    - ``precision`` (default: 20) -- an integer `p`: the error bound is at
      most `2^{-p}` on each part, plus the radii of the coefficients of `B`,
      before the rounding of the result to ``precision`` bits.

    - ``facteur``, ``processes``, ``engine`` -- as in :func:`sieve_integral`.

    - ``verbose`` (default: 0) -- if positive, print the counts of
      :func:`integrate_many`.

    OUTPUT: a ball of ``RealBallField(precision)``.

    EXAMPLES:

    On the intervals where `B(t_2/t_1)` is either `1` or `1 + \log(t_2/t_1 - 1)`::

        sage: from sieve_integral_brion.api import sieve_integral_harman
        sage: x, y = var("x, y")
        sage: polytope = [x + y == 1, 1/4 < x, x < y]
        sage: value = sieve_integral_harman(polytope, (y, x), 1 + x, precision=50); value
        [1.69211856327702 +/- ...e-15]
        sage: C = ComplexBallField(100)
        sage: exact = (C.integral(lambda t, _: (1 + t)/(t*(1 - t)), 1/3, 1/2)
        ....:          + C.integral(lambda t, a: (1 + (1/t - 2).log(analytic=a))
        ....:                                    * (1 + t)/(t*(1 - t)), 1/4, 1/3)).real()
        sage: value.overlaps(exact)
        True

    The polytope may be a Polyhedron, the Buchstab variables indices, and
    `F` a polynomial::

        sage: P = Polyhedron(eqns=[(-1, 1, 1)], ieqs=[(-1/4, 1, 0), (0, -1, 1)])
        sage: R = PolynomialRing(QQ, "x", 2)
        sage: sieve_integral_harman(P, (1, 0), 1 + R.gen(0)).overlaps(exact)
        True

    Without `B`, on `[1, 2]^2` with `F = 1 + t_1`, the integral is `(1 + \log
    2) \log 2`::

        sage: box = Polyhedron(vertices=[(1, 1), (2, 1), (1, 2), (2, 2)])
        sage: K = RealBallField(100)
        sage: value = sieve_integral_harman(box, None, 1 + R.gen(0), precision=40)
        sage: value.overlaps((1 + K(2).log()) * K(2).log())
        True
    """
    polytope, variables, border = weighted_polytope(polytope_data, facteur)
    d = polytope.ambient_dimension()
    indices = (None if buchstab_variables is None
               else variable_indices(buchstab_variables, variables))
    polynomial = parse_polynomial(integrand_polynomial_data, variables, d)
    field = RealBallField(precision)
    if border is None:
        return field.zero()
    # B >= 0, so that the weight is non-negative if F = 1
    # the essential forms of F, computed once for all the parts
    integrand = with_essential_forms(Integrand(polynomial, nonnegative=polynomial is None), border)
    if indices is None:
        return integrate_many([polytope_task(polytope, border, integrand, precision)],
                              processes, field, verbose)
    r, s = indices
    buchstab = BuchstabB(precision=precision + BUCHSTAB_EXTRA_BITS, expansion="chebyshev")
    mini, maxi = polytope.bounding_box()
    buchstab(maxi[r] / mini[s])
    ieqs, eqns, _, options = polytope_task(polytope, border, integrand, precision)
    tasks = []
    for start, end, coefficients in buchstab.chebyshev_approximants_list():
        start, end = QQ(start), QQ(end)
        # start <= t_r/t_s <= end, on which B(t_r/t_s) is the series of
        # coefficients at (2 t_r - (start + end) t_s)/(end - start)
        rows = [[0] * (d + 1) for _ in range(3)]
        rows[0][1 + r], rows[0][1 + s] = 1, -start
        rows[1][1 + r], rows[1][1 + s] = -1, end
        rows[2][1 + r], rows[2][1 + s] = 2 / (end - start), -(start + end) / (end - start)
        ratio = ChebyshevRatioFactor(tuple(coefficients), tuple(rows[2]), s)
        tasks.append((ieqs + rows[:2], eqns, dataclasses.replace(integrand, ratio=ratio), options))
    return integrate_many(tasks, processes, field, verbose)


def sieve_integral_factors(polytope_data, factors, polynomial=None, precision=20,
                           working_precision=None, facteur=None, processes=None,
                           verbose=0, nonnegative=False):
    r"""
    Return a RealBall containing `\int_P F(t) \prod_f P_f(L_f(t)) \, dt_1
    \cdots dt_k/(t_1 \cdots t_d)`, where `F` is a polynomial, the `P_f`
    are Chebyshev series and the `L_f` affine forms. The forms are assumed
    to take values in `[-1, 1]` on `P`.

    The polytope `P \subset \RR_{>0}^d` is either full-dimensional (`k =
    d`), or on the hyperplane `\sum_j t_j = 1` (`k = d - 1`), and is taken
    as one piece, or the integral is in closed form if `k \leq 1`.

    INPUT:

    - ``polytope_data`` -- a polytope, or an iterable of symbolic linear
      equalities and inequalities.

    - ``factors`` -- the pairs ``(coefficients, form)``: `P_f = \sum_n c_n
      T_n`, `c` being ``coefficients``, balls or rational numbers (the
      series of their midpoints, within the sum of their radii), and
      `L_f(t) = b + \sum_j a_j t_j`, ``form`` being `(b, a_1, \ldots, a_d)`,
      with values in `[-1, 1]` on `P`.

    - ``polynomial`` (default: ``None``, meaning `F = 1`) -- `F`, a symbolic
      expression in the variables of ``polytope_data``, or a polynomial in
      `d` variables.

    - ``precision`` (default: 20) -- the integer `p`: the error bound is at
      most `2^{-p}`, plus the radii of the coefficients of the factors.

    - ``working_precision`` (default: ``None``, meaning ``precision``) -- the
      precision of the result.

    - ``facteur``, ``processes`` -- as in :func:`sieve_integral`.

    - ``verbose`` (default: 0) -- as in :func:`sieve_integral_harman`.

    - ``nonnegative`` (default: ``False``) -- whether the integrand is
      non-negative on `P`, which halves its trivial bound.

    OUTPUT: a ball of ``RealBallField(working_precision)``.

    EXAMPLES:

    On `[4, 6]^2`, with `T_1(t_1 - 5) = t_1 - 5`, the integral is `(2 - 5
    \log(3/2)) \log(3/2)`::

        sage: from sieve_integral_brion.api import sieve_integral_factors
        sage: box = Polyhedron(vertices=[(4, 4), (6, 4), (4, 6), (6, 6)])
        sage: value = sieve_integral_factors(box, [([0, 1], (-5, 1, 0))], precision=40)
        sage: K = RealBallField(100)
        sage: value.overlaps((2 - 5 * K(3/2).log()) * K(3/2).log())
        True

    With `B(t_2/2)`, check against Arb's integration using the explicit expression of `B`
    on `[2, 3]`::

        sage: from sieve_integral_brion.dde import BuchstabB
        sage: B = BuchstabB(precision=60, expansion="chebyshev")
        sage: start, end, coefficients = B.chebyshev_approximants_list()[2]
        sage: value = sieve_integral_factors(box, [(coefficients, (-5, 0, 1))],
        ....:                                precision=40, nonnegative=True)
        sage: C = ComplexBallField(100)
        sage: integral = C.integral(lambda t, a: (1 + (t/2 - 1).log(analytic=a)) / t, 4, 6).real()
        sage: exact = K(3/2).log() * integral
        sage: value.overlaps(exact), value.rad() < 2^-39
        (True, True)

    The argument of a factor has to lie in `[-1, 1]`::

        sage: sieve_integral_factors(box, [([1], (-4, 1, 0))])
        Traceback (most recent call last):
        ...
        ValueError: the argument of a Chebyshev series leaves [-1, 1] on the polytope
    """
    polytope, variables, border = weighted_polytope(polytope_data, facteur)
    polynomial = parse_polynomial(polynomial, variables, polytope.ambient_dimension())
    field = RealBallField(precision if working_precision is None else working_precision)
    if border is None:
        return field.zero()
    chebyshev = tuple(ChebyshevAffineFactor(tuple(c), tuple(form)) for c, form in factors)
    integrand = Integrand(polynomial, chebyshev, nonnegative=nonnegative)
    return integrate_many([polytope_task(polytope, border, integrand, precision)],
                          processes, field, verbose)
