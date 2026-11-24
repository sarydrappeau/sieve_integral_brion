r"""
Check sieve_integral without cutting, on polytopes whose integral has a closed
form, and check the input handling (before integration); check that the error
bound of a piece holds against a rule of higher degree on random pieces, that
negligible pieces are replaced by their trivial bound, and that the nodes left
out by BrionSum are computed again at a higher precision.
"""
import dataclasses
import random

from sieve_integral_brion import (api as plain_api, evaluate as plain_evaluate,
                                  pieces as plain_pieces)

PLAIN_FIELD = RealBallField(200)
PLAIN_BOX = Polyhedron(vertices=[(1, 1), (2, 1), (1, 3), (2, 3)])


def test_plain_closed_forms():
    R = PLAIN_FIELD
    C = ComplexBallField(R.precision())
    simplex = Polyhedron(eqns=[(-1, 1, 1, 1)],
                         ieqs=[(-1/7, 1, 0, 0), (0, -1, 1, 0), (0, 0, -1, 1)])
    grid_box = Polyhedron(ieqs=[(-1, 1, 0), (285605/100000, -1, 0),
                                (-1, 0, 1), (6/5, 0, -1)])
    cases = [
        ("[1, 2] x [1, 3]", PLAIN_BOX, 50, R(2).log() * R(3).log()),
        ("[2, 3]^3", vector((2, 2, 2)) + polytopes.hypercube(3, intervals="zero_one"),
         50, R(3/2).log()^3),
        ("segment", Polyhedron(vertices=[(1/4, 3/4), (1/2, 1/2)]), 50, R(3).log()),
        # the docstring example of the library's sieve_integral
        ("t_1 + t_2 + t_3 = 1", simplex, 50,
         C.integral(lambda x, _: (6 - x).log() / x, 1, 5).real() / 2),
        # the library's test of the rounding of its grid
        ("grid rounding", grid_box, 30, R(285605/100000).log() * R(6/5).log()),
    ]
    for name, polytope, precision, expected in cases:
        value = plain_api.sieve_integral(polytope, precision=precision)
        check_overlap(value, expected, name)
        assert value.rad() < 2^(1 - precision), (name, value)
    # sieve_integral takes the segment in closed form: the rule on it
    piece = plain_pieces.prepare_piece(cases[2][1], True, 50)
    value = plain_evaluate.evaluate_piece(piece, RealBallField(50))
    check_overlap(value, R(3).log(), "segment, by the rule")
    assert value.rad() < 2^-49, value


def plain_rejects(arguments, options, error, message):
    try:
        plain_api.sieve_integral(*arguments, **options)
    except error as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"accepted: {arguments}, {options}")


def test_plain_inputs():
    R = PLAIN_FIELD
    sieve_integral = plain_api.sieve_integral
    check_overlap(sieve_integral(Polyhedron([(1,), (2,)])), R(2).log(), "[1, 2]")
    assert sieve_integral(Polyhedron([(1,)])) == 1
    assert sieve_integral(Polyhedron(vertices=[()])).is_zero()
    assert sieve_integral(Polyhedron(ieqs=[(-1, 1, 0), (0, -1, 0)])).is_zero()
    u, v = var("u, v")
    check_overlap(sieve_integral([u + v == 1, 1/4 < u, u < v], precision=30),
                  R(3).log(), "u + v = 1, 1/4 < u < v")
    check_overlap(sieve_integral(PLAIN_BOX, processes=1, engine="sage"),
                  R(2).log() * R(3).log(), "ignored options")
    plain_rejects((PLAIN_BOX,), {"facteur": 1.3}, NotImplementedError, "facteur")
    plain_rejects(([u + v == 1, 0 < u, u < v],), {}, ValueError, "positive orthant")
    plain_rejects((Polyhedron(eqns=[(-1, 1, 2)], ieqs=[(-1/4, 1, 0), (-1/8, 0, 1)]),),
                  {}, ValueError, "sum_i x_i = 1")


def test_plain_trivial_pieces():
    epsilon = 2^-40
    thin = Polyhedron(vertices=[(1, 1), (1 + epsilon, 1), (1, 2), (1 + epsilon, 2)])
    piece = plain_pieces.prepare_piece(thin, False, 20)
    assert piece == plain_pieces.Trivial(epsilon / 2, epsilon / 2), piece
    check_overlap(plain_api.sieve_integral(thin), PLAIN_FIELD(1 + epsilon).log()
                  * PLAIN_FIELD(2).log(), "thin box")


def test_plain_escalation():
    field = RealBallField(30)
    piece = plain_pieces.prepare_piece(PLAIN_BOX, False, 30)

    # start contains the starting node precision of the Brion sum evaluator
    # at 30 bits requested (this gives start = 128 bits)
    start = plain_evaluate.start_precision(piece, field)

    # tight is a copy of piece where the final target precision is
    # artificially raised to be larger than the starting precision of
    # the piece (in order to force an escalation of precision)
    tight = dataclasses.replace(piece, tolerance_exponent=-start - 20)

    # check that some nodes are recomputed
    assert plain_evaluate.brion_sum(tight, start).run()[1], "no node left out"
    check_overlap(plain_evaluate.evaluate_piece(tight, field),
                  plain_evaluate.evaluate_piece(piece, field), "escalated")


def test_plain_bound_audit():
    # check for stability under rule degree raising
    # |Q_N - Q_{N+20}| <= E_N + E_{N+20}, with 128 bits of precision in the rule sum
    rng = random.Random(int(6))
    checked = 0
    for d, border in ((2, False), (3, False), (2, True), (3, True), (4, True)):
        for _ in range(4):
            polytope = random_polytope(rng, d, border)    # from test_geometry.sage
            if polytope.dimension() != d - border:
                continue
            piece = plain_pieces.prepare_piece(polytope, border, 12)
            if isinstance(piece, plain_pieces.Trivial):
                continue
            low = plain_evaluate.rule_sum(piece, 128)
            # raising the node count by 10 in each direction raises the degree by 20
            config = dataclasses.replace(piece.config,
                                         ms=tuple(m + 10 for m in piece.config.ms))
            high = plain_evaluate.rule_sum(piece, 128, config)
            high_error = (2 * piece.prefactor * piece.volume
                          * plain_pieces.truncation_bound(d, piece.rate, piece.degree + 20, [1]))
            difference = abs(low - high).lower()
            assert difference <= piece.error + high_error, (
                polytope.vertices_list(), difference, piece.error)
            checked += 1
    assert checked >= 15, checked


TESTS = {
    "plain/closed-forms": (test_plain_closed_forms, ("quick",)),
    "plain/inputs": (test_plain_inputs, ("quick",)),
    "plain/trivial-pieces": (test_plain_trivial_pieces, ("quick",)),
    "plain/escalation": (test_plain_escalation, ("quick",)),
    "plain/bound-audit": (test_plain_bound_audit, ("quick",)),
}
