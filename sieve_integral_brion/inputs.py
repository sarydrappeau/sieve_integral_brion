r"""
Reading a polytope: constraints given by symbolic inequalities, and
checks that the polytope satisfies the necessary hypotheses for the
method to work.

The functions are those of the ``sieve_integral`` library
(``code/sieve_integral.py``), except :func:`as_polytope`.

EXAMPLES::

    sage: from sieve_integral_brion.inputs import *
    sage: u, v = var("u, v")
    sage: symbolic_to_eqns([u + v == 1, 1/4 < u, u < v])
    ([(-1, 1, 1)], [(-1/4, 1, 0), (0, -1, 1)], [u, v])

AUTHORS:

- Sary Drappeau, Adrien Mounier

AI DISCLOSURE:

As for ``sieve_integral.py``; Claude Opus 5.5 copied the functions here,
and wrote :func:`as_polytope`.
"""
from operator import le, ge, lt, gt, eq

from sage.geometry.polyhedron.constructor import Polyhedron
from sage.rings.rational_field import QQ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.geometry.polyhedron.base import Polyhedron_base
from sage.symbolic.expression import Expression

__all__ = ["expression_to_polynomial", "polynomial_to_equation", "symbolic_to_eqns",
           "eqn_sum_one", "is_polytope_on_sum1", "check_positive_orthant",
           "as_polytope"]


def expression_to_polynomial(expression, variables):
    r"""
    Convert a symbolic expression to a rational polynomial.

    INPUT:

    - ``expression`` -- symbolic expression

    - ``variables`` -- a list or tuple of variables containing all
    variables in `expression`

    OUTPUT: a tuple of

    - a rational polynomial,

    - generators of its polynomial ring, in the same order as the
    variables.

    EXAMPLES::

        sage: from sieve_integral_brion.inputs import expression_to_polynomial
        sage: var("a, b, c")
        (a, b, c)
        sage: expression_to_polynomial(c^2 - a + 2*b + 4, (a, b, c))
        (x2^2 - x0 + 2*x1 + 4, (x0, x1, x2))
        sage: expression_to_polynomial(c^2 - a + 2*b + 4, (c, b, a))
        (x0^2 + 2*x1 - x2 + 4, (x0, x1, x2))

    The expression must be polynomial, with rational coefficients::

        sage: expression_to_polynomial(sin(a) + b, (a, b))
        Traceback (most recent call last):
        ...
        TypeError: The expression to be converted does not coerce to a
        rational polynomial.

    and all of its variables must be listed::

        sage: expression_to_polynomial(a^2 + b - c, (a, b))
        Traceback (most recent call last):
        ...
        ValueError: Expression contains variables not in the variable list
    """
    if not set(variables).issuperset(set(expression.variables())):
        raise ValueError("Expression contains variables not in the "
                         "variable list")
    ring = PolynomialRing(QQ, len(variables), "x")
    generators = ring.gens()
    try:
        polynomial = (
            ring(expression.subs({variables[i]: generators[i]
                                  for i in range(len(variables))}))
        )
    except TypeError as exc:
        raise TypeError("The expression to be converted does not "
                        "coerce to a rational polynomial.")
    return (polynomial, generators)


def polynomial_to_equation(polynomial, generators):
    r"""
    Return the coefficients which represent the inequality
    ``polynomial >= 0``.

    INPUT:

    - ``polynomial``: a polynomial of degree at most 1.

    - ``generators``: a list of tuple of generators of the parent ring
    of ``polynomial``.

    OUTPUT: a tuple ``(a_0, a_1, ... a_k)`` where ``a_0`` is the
    constant coefficient of p, and ``a_i`` are the coefficient of the
    ``i``-th generator in ``polynomial``.

    EXAMPLES::

        sage: from sieve_integral_brion.inputs import polynomial_to_equation
        sage: R.<x, y, z, t> = PolynomialRing(QQ)
        sage: generators = R.gens(); generators
        (x, y, z, t)
        sage: polynomial_to_equation(x + 3*y - z + 3, generators)
        (3, 1, 3, -1, 0)
        sage: polynomial_to_equation(x + y + z + t - 1, generators)
        (-1, 1, 1, 1, 1)

    The polynomial has to be of degree at most one::

        sage: polynomial_to_equation(x^2 + y, generators)
        Traceback (most recent call last):
        ...
        ValueError: The polynomial should be affine linear.
    """
    if polynomial.degree() > 1:
        raise ValueError("The polynomial should be affine linear.")
    return ((polynomial.constant_coefficient(),)
            + tuple(polynomial.monomial_coefficient(x) for x in generators))


def symbolic_to_eqns(expressions):
    r"""
    Convert an iterable of symbolic linear equalities or equalities
    into a list of coefficients suitable for use in Polyhedron.

    INPUT:

    - ``expressions``: an iterable (list, tuple or set for instance)
    of symbolic linear equalities or inequalities.

    OUTPUT: a tuple ``(eqns, ieqs, variables)`` consisting of

    - ``eqns``: a list of equalities, where an equality
    `Ax + b = 0` is represented by a list ``(b, A)``

    - ``ieqs``: similarly, a list of inequalities, with
    `Ax + b \geq 0` represented by a list ``(b, A)``

    - ``variables``: the list of symbolic variables sorted by name, so
    that the ``i``-th element of ``variables`` corresponds to the
    ``i``-th coordinate in the equalities and inequalities.

    The rows follow the order in which ``expressions`` is iterated: pass
    a list rather than a set if that order matters to you.

    EXAMPLES::

        sage: from sieve_integral_brion.inputs import symbolic_to_eqns
        sage: var("a, b, c")
        (a, b, c)
        sage: symbolic_to_eqns([2*a + b - c <= 0, a > c, b + a == 3*c])
        ([(0, 1, 1, -3)], [(0, -2, -1, 1), (0, 1, 0, -1)], [a, b, c])
        sage: symbolic_to_eqns([a == 2*b, b == 2*a, a + b >= 0])
        ([(0, 1, -2), (0, -2, 1)], [(0, 1, 1)], [a, b])

    The coefficients have to be rational::

        sage: symbolic_to_eqns([sqrt(2) * a >= b])
        Traceback (most recent call last):
        ...
        TypeError: The expression to be converted does not coerce to a
        rational polynomial.

    and the constraints affine linear::

        sage: symbolic_to_eqns([a^2 <= b, a == 0, b == 0])
        Traceback (most recent call last):
        ...
        ValueError: It is likely that one of the inequalities provided
        involves non-linear terms.

    .. SEEALSO::

        :meth:`sage.geometry.polyhedron.constructor.Polyhedron`
    """
    ieqs = []
    eqns = []
    variables = set()
    for expression in expressions:
        if not isinstance(expression, Expression):
            raise ValueError(
                "Expected a symbolic equality or inequality, "
                f"got {expression!r}."
            )
        if expression.operator() not in (le, ge, lt, gt, eq):
            raise ValueError(
                f"The expression {expression!r} is not an equality "
                "or an inequality."
            )
        variables.update(expression.variables())
    variables = sorted(variables, key=str)
    for expression in expressions:
        polynomial, generators = expression_to_polynomial(
            expression.lhs() - expression.rhs(),
            variables
        )

        if expression.operator() in (le, lt):
            polynomial = -polynomial
        try:
            if expression.operator() == eq:
                eqns.append(polynomial_to_equation(polynomial, generators))
            else:
                ieqs.append(polynomial_to_equation(polynomial, generators))
        except ValueError as exc:
            raise ValueError(
                "It is likely that one of the inequalities provided "
                "involves non-linear terms."
            ) from exc
    return (eqns, ieqs, variables)


def eqn_sum_one(dim):
    r"""
    Return the inequality `\sum_{i=1}^d x_i = 1` in the Polyhedron
    ``(b, A)`` format.

    INPUT:

    - ``dim`` -- the dimension

    OUTPUT:

    - the tuple ``(-1, 1, ..., 1)``, where ``1`` is repeated ``dim`` times.

    EXAMPLES::

        sage: from sieve_integral_brion.inputs import eqn_sum_one
        sage: eqn_sum_one(4)
        (-1, 1, 1, 1, 1)
        sage: polytope = Polyhedron(eqns = [eqn_sum_one(2)])
        sage: polytope
        A 1-dimensional polyhedron in QQ^2 defined as the convex hull of
        1 vertex and 1 line
        sage: polytope.equations()
        (An equation (1, 1) x - 1 == 0,)
        sage: Polyhedron(eqns = [eqn_sum_one(0)])
        The empty polyhedron in QQ^0

    The dimension has to be non-negative::

        sage: eqn_sum_one(-2)
        Traceback (most recent call last):
        ...
        ValueError: The dimension should be non-negative.
    """
    if dim < 0:
        raise ValueError("The dimension should be non-negative.")
    return (-1,) + (1,) * dim


def is_polytope_on_sum1(polytope):
    r"""
    Check if the polytope is of the hyperplane `\sum_i x_i = 1`, or if it
    is full-dimensional.

    Raises an exception if none of the two cases occurs.

    INPUT: ``polytope`` (Polyhedron object) -- a polytope

    OUTPUT: a boolean.

    Returns True if ``polytope`` is in the hyperplane `\sum_i x_i = 1` and
    is of full-dimensional inside it, and False if ``polytope`` is full-
    dimensional.

    EXAMPLES::

        sage: from sieve_integral_brion.inputs import is_polytope_on_sum1
        sage: is_polytope_on_sum1(polytopes.hypercube(4))
        False
        sage: is_polytope_on_sum1(polytopes.simplex(4))
        True

    Any other polytope cut out by an equation is rejected::

        sage: polytope_line = Polyhedron(eqns = [(-1, 1, 0, 0), (0, 0, 1, -1)])
        sage: is_polytope_on_sum1(polytope_line)
        Traceback (most recent call last):
        ...
        ValueError: This method is currently handling only polytopes which are
        either full-dimensional, or cut out by the specific equation
        `\sum_i x_i = 1`.

    as is anything which is not a polytope at all::

        sage: is_polytope_on_sum1([1, 2])
        Traceback (most recent call last):
        ...
        TypeError: The polytope should be a Polyhedron object.
    """
    if not isinstance(polytope, Polyhedron_base):
        raise TypeError("The polytope should be a Polyhedron object.")

    if len(polytope.equations_list()) not in (0, 1):
        raise ValueError(
            "This method is currently handling only polytopes which are "
            "either full-dimensional, or cut out by the specific equation "
            r"`\sum_i x_i = 1`."
        )

    if len(polytope.equations_list()) == 0:
        return False

    eqn = polytope.equations_list()[0]
    if all(x == -eqn[0] for x in eqn[1:]):
        return True
    raise ValueError(
        "This method is currently handling only polytopes which are "
        "either full-dimensional, or cut out by the specific equation "
        r"`\sum_i x_i = 1`."
    )


def check_positive_orthant(polytope):
    r"""
    Raise an exception unless the polytope lies in the open positive
    orthant `{\mathbb R}_{>0}^k`. Positivity is a hypothesis of the
    method.

    INPUT: ``polytope`` (Polyhedron object) -- a non-empty polytope.

    OUTPUT: ``None``; an exception is raised if the condition fails.

    EXAMPLES::

        sage: from sieve_integral_brion.inputs import check_positive_orthant
        sage: check_positive_orthant(Polyhedron([(1, 1), (2, 3)]))

    A polytope touching a coordinate hyperplane is rejected::

        sage: check_positive_orthant(polytopes.simplex(2))
        Traceback (most recent call last):
        ...
        ValueError: The polytope must lie in the open positive orthant, which
        is a hypothesis of the method, but its coordinate number 0 goes down
        to 0.

    as is one extending to negative coordinates::

        sage: check_positive_orthant(polytopes.hypercube(3))
        Traceback (most recent call last):
        ...
        ValueError: The polytope must lie in the open positive orthant, which
        is a hypothesis of the method, but its coordinate number 0 goes down
        to -1.
    """
    mini = polytope.bounding_box()[0]
    for j, lower_bound in enumerate(mini):
        if lower_bound <= 0:
            raise ValueError(
                "The polytope must lie in the open positive orthant, "
                "which is a hypothesis of the method, but its coordinate "
                f"number {j} goes down to {lower_bound}."
            )


def as_polytope(polytope_data):
    r"""
    Return the polytope described by ``polytope_data``, over ``QQ``, and its
    variables.

    INPUT:

    - ``polytope_data`` -- a polytope, or an iterable of symbolic linear
      equalities and inequalities (:func:`symbolic_to_eqns`).

    OUTPUT: a pair: the polytope, and the variables of the coordinates, sorted
    by name, or ``None`` if ``polytope_data`` is a polytope.

    EXAMPLES::

        sage: from sieve_integral_brion.inputs import as_polytope
        sage: u, v = var("u, v")
        sage: polytope, variables = as_polytope([u + v == 1, 1/4 < u, u < v])
        sage: sorted(polytope.vertices_list()), variables
        ([[1/4, 3/4], [1/2, 1/2]], [u, v])
        sage: polytope, variables = as_polytope(polytopes.hypercube(2))
        sage: polytope.base_ring(), variables
        (Rational Field, None)
    """
    if isinstance(polytope_data, Polyhedron_base):
        return Polyhedron(polytope_data, base_ring=QQ), None
    eqns, ieqs, variables = symbolic_to_eqns(polytope_data)
    return Polyhedron(ieqs=ieqs, eqns=eqns, base_ring=QQ), variables
