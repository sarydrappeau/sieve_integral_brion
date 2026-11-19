r"""
Check that the sum of BrionSum over the conical product rule approaches
integrals of closed form along each of its paths: no weight, a polynomial
weight (with exponents in a box or not), the factor `Q(\psi(t)/t_s)`, and
both, on full-dimensional polytopes and on the hyperplane `\sum_j t_j = 1`.
Check also that the nodes split into chunks, and that the nodes where the
integrand is singular, or too wide, are returned.
"""
import gc
import itertools

import sieve_integral_brion
from sieve_integral_brion import geometry as brion_geometry, rules as brion_rules

BrionSum = sieve_integral_brion.brion_sum.BrionSum
BRION_SUM_FIELD = RealBallField(128)


def brion_sum_exponents(F, D):
    r"""
    The exponents of `F` variables of total degree at most `D`, by
    increasing degree.
    """
    return sorted((beta for beta in itertools.product(range(D + 1), repeat=F)
                   if sum(beta) <= D), key=sum)


def brion_sum_derivative(polynomial, beta):
    r"""
    The derivative `\partial^\beta` of ``polynomial``.
    """
    for y, e in zip(polynomial.parent().gens(), beta):
        for _ in range(e):
            polynomial = polynomial.derivative(y)
    return polynomial


def brion_sum_data(polytope, weight=None, ratio=None):
    r"""
    The data of BrionSum for `\int_P \Phi(L(t)) Q(\psi(t)/t_s) dt/\prod_j
    t_j`, the centres those of the bounding box, and the prefactor.

    - ``weight`` -- ``None``, or ``(Phi, forms)``: a polynomial in `F`
      variables and `F` affine forms ``(b, A)``, `L_f(t) = b + \langle A,
      t\rangle`.

    - ``ratio`` -- ``None``, or ``(q, s, (psi0, psi1))``: the coefficients
      of `Q`, the position of the divisor, and `\psi(t) = \psi_0 + \langle
      \psi_1, t\rangle`.
    """
    border = bool(polytope.equations_list())
    mini, maxi = polytope.bounding_box()
    c = [(a + b) / 2 for a, b in zip(mini, maxi)]
    vertices, cones = brion_geometry.vertex_cones(polytope, c, border)
    generators, cones = brion_geometry.generator_table(cones)

    def scaled(b, A, point):
        return b + sum(a * cj * x for a, cj, x in zip(A, c, point))

    data = {"vertices": vertices, "generators": generators, "cones": cones,
            "shift": int(border)}
    if weight is not None:
        Phi, forms = weight
        exponents = brion_sum_exponents(len(forms), Phi.degree())
        data["exponents"] = exponents
        data["slopes"] = [[scaled(0, A, g) for _, A in forms] for g in generators]
        data["derivatives"] = [
            [brion_sum_derivative(Phi, beta)(*[scaled(b, A, v) for b, A in forms])
             for beta in exponents]
            for v in vertices]
    if ratio is not None:
        q, s, (psi0, psi1) = ratio
        # psi(c s)/c_s, divided by s_s, is psi(t)/t_s
        data["ratio"] = (q, s, [scaled(0, psi1, g) / c[s] for g in generators],
                         [scaled(psi0, psi1, v) / c[s] for v in vertices])
    return data, (1 / c[-1] if border else 1)


def brion_sum_value(polytope, m, prec=128, weight=None, ratio=None, order=None):
    r"""
    The integral by BrionSum, with `m` nodes in each direction.
    """
    data, prefactor = brion_sum_data(polytope, weight, ratio)
    d = polytope.ambient_dimension()
    vertices, generators, cones = data.pop("vertices"), data.pop("generators"), data.pop("cones")
    brion_sum = BrionSum(prec, brion_rules.conical_factors([m] * (d - 1), prec),
                         vertices, generators, cones, data.pop("shift"), order=order, **data)
    value, failed = brion_sum.run()
    assert not failed, failed
    return prefactor * value


def segment_integral(f):
    r"""
    `\int_{1/4}^{1/2} f(t) dt`, by Arb.
    """
    return ComplexBallField(BRION_SUM_FIELD.precision()).integral(
        lambda t, _: f(t), 1/4, 1/2).real()


def test_closed_forms():
    R = BRION_SUM_FIELD
    box = Polyhedron(vertices=[(1, 1), (2, 1), (1, 3), (2, 3)])
    segment = Polyhedron(vertices=[(1/4, 3/4), (1/2, 1/2)])
    Y = PolynomialRing(QQ, 1, "y")
    y = Y.gen()
    Y2 = PolynomialRing(QQ, 2, "y")
    y0, y1 = Y2.gens()
    t1, t2 = [(0, (1, 0))], [(0, (0, 1))]
    cases = [
        ("box", box, {}, R(2).log() * R(3).log()),
        ("segment", segment, {}, R(3).log()),
        ("box, t_1", box, {"weight": (y, t1)}, R(3).log()),
        ("box, t_1^2 t_2", box, {"weight": (y0^2 * y1, t1 + t2)}, R(3)),
        ("box, (t_1 + t_2)^2", box, {"weight": ((y0 + y1)^2, t1 + t2)},
         3/2 * R(3).log() + 4 + 4 * R(2).log()),
        ("box, t_1/t_2", box, {"ratio": ([0, 1], 1, (0, (1, 0)))}, R(2/3)),
        ("box, t_1 t_1/t_2", box, {"weight": (y, t1), "ratio": ([0, 1], 1, (0, (1, 0)))},
         R(1)),
        ("segment, t_1/t_2", segment, {"ratio": ([0, 1], 1, (0, (1, 0)))}, R(2/3)),
        ("segment, 1 + t_1/t_2 + (t_1/t_2)^2", segment,
         {"ratio": ([1, 1, 1], 1, (0, (1, 0)))},
         segment_integral(lambda t: (1 + t/(1 - t) + (t/(1 - t))^2) / (t * (1 - t)))),
        ("segment, t_1^2", segment, {"weight": (y^2, t1)}, R(3/2).log() - 1/4),
        ("segment, t_1 (1 + 2 t_1/t_2)", segment,
         {"weight": (y, t1), "ratio": ([1, 2], 1, (0, (1, 0)))},
         segment_integral(lambda t: (1 + 2 * t/(1 - t)) / (1 - t))),
    ]
    for name, polytope, options, expected in cases:
        value = brion_sum_value(polytope, 20, **options)
        assert abs(value - expected) < 10^-12, (name, value, expected)


def test_dimension_three():
    R = BRION_SUM_FIELD
    box = Polyhedron(vertices=list(itertools.product((1, 2), (1, 3), (2, 5))))
    expected = R(2).log() * R(3).log() * R(5/2).log()
    for order in (None, [2, 0, 1], [1, 2, 0]):
        value = brion_sum_value(box, 16, order=order)
        assert abs(value - expected) < 10^-12, (order, value)
    # the docstring example of the library's sieve_integral
    simplex = Polyhedron(eqns=[(-1, 1, 1, 1)],
                         ieqs=[(-1/7, 1, 0, 0), (0, -1, 1, 0), (0, 0, -1, 1)])
    value = brion_sum_value(simplex, 24)
    assert abs(value - R(0.956913527101737)) < 10^-12, value


def test_chunks():
    box = Polyhedron(vertices=list(itertools.product((1, 2), (1, 3), (2, 5))))
    data, _ = brion_sum_data(box)
    brion_sum = BrionSum(80, brion_rules.conical_factors([5, 4], 80), data["vertices"],
                         data["generators"], data["cones"], 0)
    assert brion_sum.nodes_count == (5, 4)
    whole, _ = brion_sum.run()
    parts = [brion_sum.run(lo, hi)[0] for lo, hi in ((0, 2), (2, 3), (3, 5))]
    assert sum(parts).overlaps(whole), (parts, whole)
    everything = list(itertools.product(range(5), range(4)))
    assert brion_sum.run(explicit=everything)[0].overlaps(whole)
    assert brion_sum.run(5, 5)[0].is_zero()


def test_failed_nodes():
    # the edge e_1 - e_2 with equal centres: a_g = 0 at the middle node
    segment = Polyhedron(vertices=[(1/4, 3/4), (3/4, 1/4)])
    data, _ = brion_sum_data(segment)
    assert data["generators"] == [(1, -1)]
    for m, expected in ((3, [(1,)]), (4, [])):
        brion_sum = BrionSum(64, brion_rules.conical_factors([m], 64), data["vertices"],
                             data["generators"], data["cones"], 1)
        assert brion_sum.run()[1] == expected, (m, brion_sum.run())
    box = Polyhedron(vertices=[(1, 1), (2, 1), (1, 3), (2, 3)])
    data, _ = brion_sum_data(box)
    brion_sum = BrionSum(64, brion_rules.conical_factors([6], 64), data["vertices"],
                         data["generators"], data["cones"], 0, tolerance_exponent=-200)
    value, failed = brion_sum.run()
    assert value.is_zero() and failed == [(i,) for i in range(6)], (value, failed)


def brion_sum_rejects(arguments, options, message):
    try:
        BrionSum(*arguments, **options)
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"accepted: {options}")


def test_validation():
    box = Polyhedron(vertices=[(1, 1), (2, 1), (1, 3), (2, 3)])
    data, _ = brion_sum_data(box)
    factors = brion_rules.conical_factors([3], 64)
    arguments = (64, factors, data["vertices"], data["generators"], data["cones"], 0)
    slopes = [[1] for _ in data["generators"]]
    derivatives = [[1, 1] for _ in data["vertices"]]
    brion_sum_rejects(arguments, {"exponents": [(0,), (2,)], "slopes": slopes,
                                  "derivatives": derivatives}, "predecessors")
    brion_sum_rejects(arguments, {"exponents": [(1,), (0,)], "slopes": slopes,
                                  "derivatives": derivatives}, "start with 0")
    brion_sum_rejects(arguments, {"exponents": [(0,), (1,)], "derivatives": derivatives},
                      "slopes")
    brion_sum_rejects(arguments, {"order": [0, 0]}, "permutation")
    cones = data["cones"]
    brion_sum_rejects((64, factors, data["vertices"], data["generators"],
                       [cones[0], cones[1], cones[0]], 0), {}, "follow each other")
    brion_sum_rejects((64, [], data["vertices"], data["generators"], cones, 0), {},
                      "d >= 2")
    # a value which is not a number fails once the arrays are allocated
    vertices = [(object(), 1)] + list(data["vertices"][1:])
    try:
        BrionSum(64, factors, vertices, data["generators"], cones, 0)
    except TypeError:
        gc.collect()
    else:
        raise AssertionError("accepted a vertex which is not a number")
    brion_sum = BrionSum(*arguments)
    try:
        brion_sum.__init__(*arguments)
    except RuntimeError as exc:
        assert "already initialised" in str(exc), str(exc)
    else:
        raise AssertionError("initialised twice")


TESTS = {
    "brion_sum/closed-forms": (test_closed_forms, ("quick",)),
    "brion_sum/dimension-three": (test_dimension_three, ("quick",)),
    "brion_sum/chunks": (test_chunks, ("quick",)),
    "brion_sum/failed-nodes": (test_failed_nodes, ("quick",)),
    "brion_sum/validation": (test_validation, ("quick",)),
}
