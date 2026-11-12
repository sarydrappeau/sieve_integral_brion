r"""
The helpers copied from the ``sieve_integral`` library behave as the library's.
"""
from sieve_integral_brion import chebyshev as brion_chebyshev, inputs as brion_inputs


def helper_outcome(function, *args):
    r"""
    The value of ``function(*args)``, or the type and message of the
    exception it raises.
    """
    try:
        return function(*args)
    except Exception as exc:
        return (type(exc), str(exc))


def compare_with_library(name, module, arguments):
    r"""
    Check that ``name`` of ``module`` and of the library agree on each
    tuple of ``arguments``.
    """
    library = library_module()
    for args in arguments:
        ours = helper_outcome(getattr(module, name), *args)
        theirs = helper_outcome(getattr(library, name), *args)
        assert ours == theirs, f"{name}{args}: {ours} != {theirs}"


def test_symbolic():
    a, b, c = SR.var("a, b, c")
    compare_with_library("symbolic_to_eqns", brion_inputs, [
        ([a + b + c == 1, a > 1/7, a < b, b <= c],),
        ([2*a + b - c <= 0, a > c, b + a == 3*c],),
        ([a >= 0.345, b < 3/10*a, c == 2],),
        ([sqrt(2) * a >= b],),
        ([a^2 <= b, a == 0],),
        ([a + 1],)])
    compare_with_library("expression_to_polynomial", brion_inputs, [
        (c^2 - a + 2*b + 4, (c, b, a)), (a^2 + b - c, (a, b))])


def test_polytopes():
    polytopes_ = [polytopes.hypercube(3), polytopes.simplex(3),
                  Polyhedron(eqns=[(-1, 1, 1, 1)], ieqs=[(-1/7, 1, 0, 0),
                                                        (0, -1, 1, 0), (0, 0, -1, 1)]),
                  Polyhedron(eqns=[(-1, 1, 0, 0), (0, 0, 1, -1)]),
                  Polyhedron([(1, 1), (2, 3)]), [1, 2]]
    compare_with_library("is_polytope_on_sum1", brion_inputs, [(p,) for p in polytopes_])
    compare_with_library("check_positive_orthant", brion_inputs, [(p,) for p in polytopes_[:-1]])
    compare_with_library("eqn_sum_one", brion_inputs, [(0,), (4,), (-2,)])


def test_chebyshev():
    compare_with_library("chebyshev_inverse_rate", brion_chebyshev,
            [(QQ(k) / 97, bits) for k in range(-1, 99) for bits in (2, 8, 12)]
            + [(1 - QQ(10)**-9,)])
    compare_with_library("chebyshev_inverse_powers", brion_chebyshev,
            [(t, 6, 30) for t in (QQ(1)/3, QQ(269)/4096, QQ(683)/2048, QQ(4001)/4096)])
    compare_with_library("chebyshev_affine_composition", brion_chebyshev,
            [([1, -2, 3, 5/7], -1/3, 2/3), ([0, 0, 1], 1/2, 1/4),
             ([QQ(k)**2 / 13 for k in range(12)], 3/8, 5/9)])


TESTS = {
    "helpers/inputs/symbolic": (test_symbolic, ("quick",)),
    "helpers/inputs/polytopes": (test_polytopes, ("quick",)),
    "helpers/chebyshev": (test_chebyshev, ("quick",)),
}
