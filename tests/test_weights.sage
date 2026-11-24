r"""
Check sieve_integral_harman and sieve_integral_factors, without cutting, on
integrals of closed form or given by one-dimensional integrals in Arb:
polynomials `F` of degree up to 3, full-dimensional or on `\sum_j t_j = 1`,
one and two Chebyshev factors, and Harman's factor `B(t_r/t_s)`. Check also
their inputs, and that the truncation bound with a weight holds against a
rule of higher degree on random pieces.
"""
import random

from sieve_integral_brion import (api as weights_api, dde as weights_dde,
                                  evaluate as weights_evaluate,
                                  integrand as weights_integrand,
                                  pieces as weights_pieces)

WEIGHTS_FIELD = RealBallField(200)


def weights_integral(f, a, b):
    r"""
    `\int_a^b f(t) dt`, by Arb.
    """
    C = ComplexBallField(WEIGHTS_FIELD.precision())
    return C.integral(lambda t, _: f(t), a, b).real()


def weights_check(value, expected, precision, what, parts=1):
    r"""
    Check that ``value`` overlaps ``expected``, with a radius below `4 \cdot
    2^{-p}` for each of ``parts``: `2^{-p}` for the piece, and the roundings
    to `p` bits.
    """
    check_overlap(value, expected, what)
    assert value.rad() < parts * 2^(2 - precision), (what, value)


def test_weights_polynomials():
    R = WEIGHTS_FIELD
    harman = weights_api.sieve_integral_harman
    box = Polyhedron(vertices=[(1, 1), (2, 1), (1, 2), (2, 2)])
    wide = Polyhedron(vertices=[(1, 1), (2, 1), (1, 3), (2, 3)])
    x1, x2 = PolynomialRing(QQ, "x", 2).gens()
    cases = [
        ("[1, 2]^2, 1 + t_1", box, 1 + x1, (1 + R(2).log()) * R(2).log()),
        ("[1, 2]^2, t_1^2 t_2", box, x1^2 * x2, R(3/2)),
        ("[1, 2]^2, (t_1 + t_2)^2", box, (x1 + x2)^2, 2 + 3 * R(2).log()),
        ("[1, 2] x [1, 3], t_1^2 t_2 + 3 t_1 + t_2^3", wide, x1^2 * x2 + 3 * x1 + x2^3,
         3 + 3 * R(3).log() + 26/3 * R(2).log()),
    ]
    # the triangle t_j >= 1/4 of t_1 + t_2 + t_3 = 1, where t_2 runs from 1/4
    # to 3/4 - t_1
    triangle = Polyhedron(vertices=[(1/2, 1/4, 1/4), (1/4, 1/2, 1/4), (1/4, 1/4, 1/2)])
    t1, t2, t3 = PolynomialRing(QQ, "t", 3).gens()
    cases += [
        ("triangle, t_3^2", triangle, t3^2, weights_integral(
            lambda t: ((1 - t) * (3 - 4 * t).log() - (1/2 - t)) / t, 1/4, 1/2)),
        ("triangle, t_1 t_2 t_3", triangle, t1 * t2 * t3, R(1/32)),
    ]
    for name, polytope, F, expected in cases:
        weights_check(harman(polytope, None, F, precision=50), expected, 50, name)


def test_weights_harman():
    harman = weights_api.sieve_integral_harman
    x, y = var("x, y")
    polytope = [x + y == 1, 1/4 < x, x < y]
    for name, F, f in [("1 + x", 1 + x, lambda t: 1 + t), ("x^2", x^2, lambda t: t^2),
                       ("x^3", x^3, lambda t: t^3)]:
        # B(y/x) is 1 for x in [1/3, 1/2], and 1 + log(y/x - 1) for x in [1/4, 1/3]
        expected = (weights_integral(lambda t: f(t) / (t * (1 - t)), 1/3, 1/2)
                    + weights_integral(lambda t: (1 + (1/t - 2).log()) * f(t) / (t * (1 - t)),
                                       1/4, 1/3))
        for precision in (20, 50):
            weights_check(harman(polytope, (y, x), F, precision=precision), expected,
                          precision, f"harman, {name}, {precision} bits", parts=2)
    value = harman(polytope, (y, x), 1 + x, precision=50)
    assert str(value).startswith("[1.69211856327702 +/-"), value


def test_weights_factors():
    R = WEIGHTS_FIELD
    factors = weights_api.sieve_integral_factors
    box = Polyhedron(vertices=[(4, 4), (6, 4), (4, 6), (6, 6)])
    log = R(3/2).log()
    t1, t2 = PolynomialRing(QQ, "x", 2).gens()
    B = weights_dde.BuchstabB(precision=60, expansion="chebyshev")
    on_2_3 = B.chebyshev_approximants_list()[2][2]
    # B(t/2) on [4, 6], where t/2 - 1 is in [1, 2]
    B_integral = weights_integral(lambda t: (1 + (t/2 - 1).log()) / t, 4, 6)
    cases = [
        ("T_1(t_1 - 5)", [([0, 1], (-5, 1, 0))], None, (2 - 5 * log) * log),
        ("T_1(t_1 - 5) T_2(t_2 - 5)", [([0, 1], (-5, 1, 0)), ([0, 0, 1], (-5, 0, 1))], None,
         (2 - 5 * log) * (-20 + 49 * log)),
        ("B(t_2/2)", [(on_2_3, (-5, 0, 1))], None, log * B_integral),
        ("t_1 B(t_2/2)", [(on_2_3, (-5, 0, 1))], t1, 2 * B_integral),
    ]
    for name, chebyshev, F, expected in cases:
        weights_check(factors(box, chebyshev, F, precision=40), expected, 40, name)
    value = factors(box, [(on_2_3, (-5, 0, 1))], precision=30, working_precision=60)
    assert value.parent() is RealBallField(60), value.parent()
    check_overlap(value, log * B_integral, "working precision")


def weights_rejects(function, arguments, options, error, message):
    try:
        function(*arguments, **options)
    except error as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"accepted: {arguments}, {options}")


def test_weights_inputs():
    harman = weights_api.sieve_integral_harman
    factors = weights_api.sieve_integral_factors
    x, y = var("x, y")
    box = Polyhedron(vertices=[(1, 1), (2, 1), (1, 2), (2, 2)])
    assert harman(Polyhedron(ieqs=[(-1, 1, 0), (0, -1, 0)]), (1, 0), None).is_zero()
    assert factors([x + y == 1, x > 2, y > 0], []).is_zero()
    weights_rejects(harman, ([x + y == 1, 1/4 < x, x < y], (x,)), {}, ValueError,
                    "two elements")
    weights_rejects(harman, (box, (1, 0), 1 + x), {}, ValueError, "symbolic integrand")
    weights_rejects(harman, (box, (1, 0)), {"facteur": 1.3}, NotImplementedError, "facteur")
    weights_rejects(harman, (Polyhedron([(1,), (2,)]), None), {}, NotImplementedError,
                    "dimension")
    weights_rejects(factors, (box, [([1], (-4, 1, 0))]), {}, ValueError, "leaves [-1, 1]")
    weights_rejects(factors, (box, [([1], (0, 1))]), {}, ValueError, "3 coefficients")


def test_weights_bound_audit():
    # |Q_N - Q_{N+20}| <= E_N + E_{N+20}, the rule summed at 128 bits, with
    # a polynomial of degree 2 and the ratio of a Chebyshev series
    rng = random.Random(int(7))
    ratio_series = (QQ(1), QQ(1/2), QQ(-1/4), QQ(1/8))
    checked = 0
    for d, border in ((2, False), (3, False), (2, True), (3, True)):
        ring = PolynomialRing(QQ, "t", d)
        for _ in range(4):
            polytope = random_polytope(rng, d, border)    # from test_geometry.sage
            if polytope.dimension() != d - border:
                continue
            t = ring.gens()
            F = sum(QQ(rng.randint(-3, 3)) * t[rng.randrange(d)] * t[rng.randrange(d)]
                    for _ in range(3)) + 1
            values = [v[0] / v[-1] for v in polytope.vertices_list()]
            low, high = min(values), max(values)
            form = [0] * (d + 1)
            form[1], form[d] = 2 / (high - low), -(low + high) / (high - low)
            integrand = weights_integrand.Integrand(
                F, ratio=weights_integrand.ChebyshevRatioFactor(ratio_series, tuple(form), d - 1))
            piece = weights_pieces.prepare_piece(polytope, border, 12, integrand)
            if isinstance(piece, weights_pieces.Trivial):
                continue
            low_sum = weights_evaluate.rule_sum(piece, 128)
            high_sum = weights_evaluate.rule_sum(piece, 128, ms=[m + 10 for m in piece.ms])
            bounds = [weights_pieces.truncation_bound(d, piece.rate, N, piece.weights)
                      for N in (piece.degree, piece.degree + 20)]
            difference = abs(low_sum - high_sum).lower()
            assert difference <= sum(bounds), (polytope.vertices_list(), F, difference,
                                               bounds)
            checked += 1
    assert checked >= 12, checked


TESTS = {
    "weights/polynomials": (test_weights_polynomials, ("quick",)),
    "weights/harman": (test_weights_harman, ("quick",)),
    "weights/factors": (test_weights_factors, ("quick",)),
    "weights/inputs": (test_weights_inputs, ("quick",)),
    "weights/bound-audit": (test_weights_bound_audit, ("quick",)),
}
