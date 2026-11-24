r"""
Sieve integrals by the radial Brion formula.

Computes rigorous bounds, as ``RealBall`` elements, of integrals
`\int_P W(t) \, dt_1 \cdots dt_d / (t_1 \cdots t_d)` over rational polytopes
`P` of the open positive orthant, full-dimensional or on the hyperplane
`\sum_j t_j = 1`.

AUTHORS:

- Sary Drappeau

AI DISCLOSURE:

Claude Opus 5.5 wrote this package.
"""
import sage.all  # noqa: F401  (Sage must be initialised before its modules)

from ._brion_sum_loader import load_brion_sum

brion_sum = load_brion_sum()

from .api import sieve_integral, sieve_integral_factors, sieve_integral_harman  # noqa: E402
