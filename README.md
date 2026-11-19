# Sieve integrals by a Brion formula integrated along the radius

Computes rigorous approximations, as `RealBall` elements, of integrals
of the shape

$$ \int_P W(t) \frac{dt_1 \dotsb dt_d}{t_1 \dotsb t_d} $$

over rational polytopes $P$ in the open positive orthant,
full-dimensional or on the hyperplane $\sum_j t_j = 1$. The underlying
method is to write the integral as one over the standard simplex of a
function given in closed form by Brion's formula on the vertex cones
of $P$, and integrates it by a cubature rule having positive weights,
with a rigorous bound on the error.

The user-end functions follow the same pattern (parameters, return
values) as those of the
[`sieve_integral`](https://github.com/sarydrappeau/sieve_integral)
library, which computes the same type of integrals with another
argument, relying on LattE.

## Requirements

- SageMath (11.0.beta0 was used in the development process), with its Cython and FLINT.
- The module `number_theoretic_dde_solutions.py` of the `sieve_integral`
  library _op.cit._, for the Dickman and Buchstab functions. It is looked for in the
  directory named by the environment variable `SIEVE_INTEGRAL_CODE`, and
  otherwise in `../sieve_integral/code`; failing that, it is downloaded from
  the library's GitHub repository into `~/.cache/sieve_integral_brion`.

## Building

```
sage -python setup.py build_ext --inplace
```

Without this step, importing the package builds its compiled module the first time
(set `SIEVE_BRION_AUTOBUILD=0` to prevent it).

## Tests

From the repository root:

```
sage tests/run_tests.sage            # the quick tests
sage tests/run_tests.sage --long     # all of them
PYTHONPATH=$PWD sage -t --long sieve_integral_brion/*.py sieve_integral_brion/*.pyx
```

The tests read the examples, reference values and DDE module of the
`sieve_integral` library from the directory named by `SIEVE_INTEGRAL_ROOT`
(by default `../sieve_integral`).
