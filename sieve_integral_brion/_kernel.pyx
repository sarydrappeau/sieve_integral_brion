# distutils: libraries = flint
r"""
The Arb kernel of the radial Brion method.

AUTHORS:

- Sary Drappeau

AI DISCLOSURE:

Claude Opus 5.5 wrote this module.
"""
from sage.libs.flint.arb cimport arb_add, arb_zero
from sage.rings.real_arb cimport RealBall
from sage.rings.real_arb import RealBallField


def ball_sum(balls, long prec):
    r"""
    Return the sum of ``balls`` in a ``RealBallField`` of precision ``prec``.

    EXAMPLES::

        sage: from sieve_integral_brion._kernel import ball_sum
        sage: ball_sum([RBF(1), RBF(1/3)], 53)
        [1.333333333333333 +/- ...e-16]
    """
    cdef RealBall total = RealBallField(prec)(0)
    cdef RealBall ball
    arb_zero(total.value)
    for ball in balls:
        arb_add(total.value, total.value, ball.value, prec)
    return total
