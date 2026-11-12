r"""
The Gauss-Jacobi rules are certified and exact, and so is their conical
product on the simplex.
"""
import itertools

from sieve_integral_brion import rules as brion_rules


def test_gauss_jacobi():
    for m in range(1, 13):
        for a in range(8):
            q = brion_rules.jacobi_polynomial(m, a)
            nodes, weights = brion_rules.gauss_jacobi(m, a, 128)
            ends = [(x.lower().exact_rational(), x.upper().exact_rational()) for x in nodes]
            assert 0 < ends[0][0] and ends[-1][1] < 1, (m, a, ends)
            assert all(b < c for (_, b), (c, _) in zip(ends, ends[1:])), (m, a)
            assert all(q(lo) * q(hi) < 0 for lo, hi in ends), (m, a)
            assert all(w > 0 for w in weights), (m, a)
            for j in range(2 * m):
                moment = factorial(j) * factorial(a) / factorial(j + a + 1)
                value = sum(w * x^j for x, w in zip(nodes, weights))
                assert value.contains_exact(moment), (m, a, j, value, moment)


def dirichlet_moment(alpha):
    r"""
    `\int_\Delta x^\alpha dx_1 \cdots dx_{d-1} = \prod_j \alpha_j! / (|\alpha| + d - 1)!`.
    """
    return (prod(factorial(e) for e in alpha)
            / factorial(sum(alpha) + len(alpha) - 1))


def test_conical_exactness():
    for ms, order in [([3], None), ([1, 1], None), ([2, 3], None), ([3, 3], [2, 0, 1]),
                      ([4, 3, 4], None), ([2, 2, 2], [3, 1, 0, 2]), ([3, 2, 3, 3], None)]:
        rule = brion_rules.conical_rule(ms, 128, order)
        d = len(ms) + 1
        degree = 2 * min(ms) - 1
        assert len(rule) == prod(ms)
        assert all(w > 0 for _, w in rule)
        for alpha in itertools.product(range(degree + 1), repeat=d):
            if sum(alpha) > degree:
                continue
            value = sum(w * prod(x[j]^alpha[j] for j in range(d)) for x, w in rule)
            assert value.contains_exact(dirichlet_moment(alpha)), (ms, order, alpha, value)


def test_rational_roots():
    found = {(m, a): brion_rules.jacobi_rational_roots(m, a)
             for m in range(1, 13) for a in range(8)
             if brion_rules.jacobi_rational_roots(m, a)}
    expected = {(1, a): (1 / QQ(a + 2),) for a in range(8)}
    expected.update({(m, 0): (QQ(1)/2,) for m in range(3, 13, 2)})
    expected[2, 6] = (QQ(1)/15, QQ(1)/3)
    assert found == expected, found


TESTS = {
    "rules/gauss-jacobi": (test_gauss_jacobi, ("quick",)),
    "rules/conical-exactness": (test_conical_exactness, ("quick",)),
    "rules/rational-roots": (test_rational_roots, ("quick",)),
}
