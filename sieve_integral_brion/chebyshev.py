r"""
Chebyshev expansions with rational coefficients.

The functions are borrowed from the ``sieve_integral`` library (``code/sieve_integral.py``),
except :func:`kept_count`, :func:`recentre` and :func:`to_monomial`.

EXAMPLES::

    sage: from sieve_integral_brion.chebyshev import *
    sage: chebyshev_inverse_rate(3/5)
    683/2048
    sage: chebyshev_inverse_powers(1/3, 1, 3)
    [[1, 0, 0], [5/4, -5/6, 5/18]]

AUTHORS:

- Sary Drappeau, Kevin Ford, Adrien Mounier

AI DISCLOSURE:

As for ``sieve_integral.py``; Claude Opus 5.5 copied the functions here,
and wrote :func:`kept_count`, :func:`recentre` and :func:`to_monomial`.
"""
from sage.rings.rational_field import QQ
from sage.rings.real_mpfr import RR
from sage.functions.other import binomial
from sage.functions.orthogonal_polys import chebyshev_T
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing

__all__ = ["chebyshev_inverse_rate", "chebyshev_inverse_powers",
           "chebyshev_affine_composition", "kept_count", "recentre", "to_monomial"]


def chebyshev_inverse_rate(ratio, bits=12):
    r"""
    Return the smallest multiple `t` of `2^{-bits}` for which
    `r' = 2t/(1 + t^2)` is at least `r`, ``ratio``.

    On `[-1, 1]`, `1/(1 + r u) = 1/(1 + r' u')` with `u' = (r/r') u` in
    `[-1, 1]`, and the Chebyshev coefficients of `1/(1 + r' u')` are
    rational and decay like `t^n` (see :func:`chebyshev_inverse_powers`).
    The fastest decay, at `r' = r`, is `r/(1 + \sqrt{1 - r^2})`, irrational
    in general.

    INPUT:

    - ``ratio`` -- a rational number `r` with `0 < r < 1`.

    - ``bits`` (default: 12) -- the number of binary digits of `t`.

    OUTPUT: a dyadic rational `t` with `0 < t < 1`.

    EXAMPLES:

    For `r = 3/5` the fastest decay is `1/3`::

        sage: from sieve_integral_brion.chebyshev import chebyshev_inverse_rate
        sage: t = chebyshev_inverse_rate(3/5); t
        683/2048
        sage: 2*t / (1 + t^2) >= 3/5 > 2*(t - 2^-12) / (1 + (t - 2^-12)^2)
        True
        sage: chebyshev_inverse_rate(3/5, bits = 2)
        1/2

    The ratio has to lie in `(0, 1)`, and not too close to 1 for
    ``bits``::

        sage: chebyshev_inverse_rate(1)
        Traceback (most recent call last):
        ...
        ValueError: The ratio should lie in (0, 1).
        sage: chebyshev_inverse_rate(1 - 10^-9)
        Traceback (most recent call last):
        ...
        ValueError: The ratio is too close to 1 for a rate of 12 bits.
    """
    if not 0 < ratio < 1:
        raise ValueError("The ratio should lie in (0, 1).")
    step = QQ(2)**(-bits)
    # 2t/(1 + t^2) increases with t on [0, 1]: start from a floating-point
    # estimate of the fastest decay, then correct it exactly.
    fastest = RR(ratio) / (1 + (1 - RR(ratio)**2).sqrt())
    t = QQ((fastest / step).ceil()) * step
    while 2*t / (1 + t**2) < ratio:
        t += step
    while t > step and 2*(t - step) / (1 + (t - step)**2) >= ratio:
        t -= step
    if t >= 1:
        raise ValueError(f"The ratio is too close to 1 for a rate of "
                         f"{bits} bits.")
    return t


def chebyshev_inverse_powers(rate, power, count):
    r"""
    Return the Chebyshev coefficients of `(1 + r u)^{-m}` on `[-1, 1]`, for
    `m = 0, \ldots,` ``power``, where `r = 2t/(1 + t^2)` and `t` is ``rate``.

    With `u = (y + 1/y)/2`, `1 + r u = (1 + t y)(1 + t/y)/(1 + t^2)`, so
    `(1 + r u)^{-m} = \sum_{n \in \ZZ} \ell_{m,n} y^n` with
    `\ell_{m,-n} = \ell_{m,n}`, `(-t)^n` times a positive number. The
    coefficient of `T_n` is `\ell_{m,0}` for `n = 0` and `2 \ell_{m,n}`
    otherwise: it has the sign `(-1)^n`, and the absolute values add up to
    the value at `u = -1`, `(1 - r)^{-m} = ((1 + t^2)/(1 - t)^2)^m`. Euler's
    transformation of a terminating hypergeometric sum gives

    .. MATH::

        \ell_{m,0} = \frac{(1 + t^2)^m}{(1 - t^2)^{2m-1}}
                     \sum_{i < m} \binom{m-1}{i}^2 t^{2i},

    and `(1 + r u) (1 + r u)^{-m} = (1 + r u)^{-(m-1)}`, coefficient by
    coefficient, the others:
    `\ell_{m,n+1} = (2/r)(\ell_{m-1,n} - \ell_{m,n}) - \ell_{m,n-1}`. The
    computation is exact.

    INPUT:

    - ``rate`` -- a rational number `t` with `0 < t < 1`.

    - ``power`` -- a non-negative integer.

    - ``count`` -- the number of coefficients of each power.

    OUTPUT: a list of ``power + 1`` lists of ``count`` rational numbers, the
    coefficients of `T_0, T_1, \ldots` in `(1 + r u)^{-m}`.

    EXAMPLES:

    For `t = 1/3`, `r = 3/5`. The first power has the coefficients
    `\ell_{1,0} = (1 + t^2)/(1 - t^2)` and `2 \ell_{1,0} (-t)^n`::

        sage: from sieve_integral_brion.chebyshev import chebyshev_inverse_powers
        sage: powers = chebyshev_inverse_powers(1/3, 2, 4)
        sage: powers
        [[1, 0, 0, 0], [5/4, -5/6, 5/18, -5/54], [125/64, -75/32, 325/288, -425/864]]
        sage: powers[1] == [5/4] + [2 * 5/4 * (-1/3)^n for n in range(1, 4)]
        True

    Against the function, the signs and the masses::

        sage: t = 269/4096; r = 2*t / (1 + t^2)
        sage: powers = chebyshev_inverse_powers(t, 6, 40)
        sage: all(abs(sum(c * chebyshev_T(n, u) for n, c in enumerate(powers[m]))
        ....:         - 1/(1 + r*u)^m) < 10^-30
        ....:     for m in range(7) for u in (-1, -1/3, 0, 1/2, 1))
        True
        sage: all(c * (-1)^n > 0 for m in range(1, 7) for n, c in enumerate(powers[m]))
        True
        sage: all(0 < (1 + t^2)^m / (1 - t)^(2*m) - sum(abs(c) for c in powers[m]) < 10^-30
        ....:     for m in range(1, 7))
        True
    """
    r = 2 * rate / (1 + rate**2)
    laurent = [[QQ(1)] + [QQ(0)] * count]
    for m in range(1, power + 1):
        previous = laurent[-1]
        current = [(1 + rate**2)**m / (1 - rate**2)**(2*m - 1)
                   * sum(binomial(m - 1, i)**2 * rate**(2*i) for i in range(m))]
        current.append((previous[0] - current[0]) / r)
        for n in range(1, count):
            current.append(2 / r * (previous[n] - current[n]) - current[n - 1])
        laurent.append(current)
    return [[l[0]] + [2 * c for c in l[1:count]] for l in laurent]


def chebyshev_affine_composition(coefficients, shift, scale):
    r"""
    Return the Chebyshev coefficients of `P(\mu + \lambda Y)`, where
    `P = \sum_i c_i T_i`, `c` being ``coefficients``, `\mu` ``shift`` and
    `\lambda` ``scale``.

    INPUT:

    - ``coefficients`` -- a list of rational numbers.

    - ``shift``, ``scale`` -- rational numbers.

    OUTPUT: a list of rational numbers, the coefficients of
    `T_0, T_1, \ldots`, as many as ``coefficients`` has.

    EXAMPLES:

    `T_2(1/2 + Y/4) = -1/2 + Y/2 + Y^2/8 = -7/16 + T_1(Y)/2 + T_2(Y)/16`::

        sage: from sieve_integral_brion.chebyshev import chebyshev_affine_composition
        sage: chebyshev_affine_composition([0, 0, 1], 1/2, 1/4)
        [-7/16, 1/2, 1/16]
        sage: c = chebyshev_affine_composition([1, -2, 3, 5/7], -1/3, 2/3)
        sage: P = lambda a: 1 - 2*a + 3*chebyshev_T(2, a) + 5/7*chebyshev_T(3, a)
        sage: all(sum(b * chebyshev_T(n, y) for n, b in enumerate(c))
        ....:     == P(-1/3 + 2*y/3) for y in (-1, -1/5, 0, 1/2, 1))
        True
    """
    Y = PolynomialRing(QQ, "Y").gen()
    composed = sum((QQ(c) * chebyshev_T(n, shift + scale * Y)
                    for n, c in enumerate(coefficients)), Y.parent().zero())
    result = [QQ(0)] * len(coefficients)
    for i, a in enumerate(composed.list()):
        for k in range(i // 2 + 1):
            # Y^i = 2^(1-i) sum_k C(i, k) T_{i-2k}, the middle term halved
            zeta = QQ(binomial(i, k)) / 2**(i - 1) if i else QQ(1)
            if i and 2*k == i:
                zeta /= 2
            result[i - 2*k] += a * zeta
    return result


def kept_count(coefficients, allowance):
    r"""
    Return the least `K \geq 1` with `\sum_{n \geq K} |c_n| \leq`
    ``allowance``, `c` being ``coefficients``, and that sum.

    EXAMPLES::

        sage: from sieve_integral_brion.chebyshev import kept_count
        sage: kept_count([1, 1/2, 1/8, 1/16], 1/5)
        (2, 3/16)
        sage: kept_count([1, 1/2, 1/8, 1/16], 0), kept_count([1/2, 1/4], 1)
        ((4, 0), (1, 1/4))
    """
    tail = QQ(0)
    K = len(coefficients)
    while K > 1 and tail + abs(coefficients[K - 1]) <= allowance:
        tail += abs(coefficients[K - 1])
        K -= 1
    return K, tail


def recentre(low, high, step=QQ(2)**-12):
    r"""
    Return multiples `\mu` and `\lambda > 0` of ``step`` such that `[\mu -
    \lambda, \mu + \lambda]` contains the interval from ``low`` to ``high``.

    EXAMPLES::

        sage: from sieve_integral_brion.chebyshev import recentre
        sage: recentre(1/3, 1/2, 1/8)
        (3/8, 1/8)
        sage: recentre(1/3, 1/3)
        (1365/4096, 1/4096)
    """
    mu = ((low + high) / 2 / step).round() * step
    lam = max(step, (max(high - mu, mu - low) / step).ceil() * step)
    return mu, lam


def to_monomial(coefficients):
    r"""
    Return `\sum_n c_n T_n(y)`, `c` being ``coefficients``, as a polynomial
    over ``QQ``.

    EXAMPLES::

        sage: from sieve_integral_brion.chebyshev import to_monomial
        sage: to_monomial([1, 1/2, 1/4])
        1/2*y^2 + 1/2*y + 3/4
    """
    y = PolynomialRing(QQ, "y").gen()
    return sum((QQ(c) * chebyshev_T(n, y) for n, c in enumerate(coefficients)),
               y.parent().zero())
