r"""
Check the configurations of the rule: the pairs of coordinates along which
a piece has edges, the order and numbers of nodes which keep the nodes off
the hyperplanes `\langle x, g\rangle = 0`, the detection of singular nodes
and the reconfigurations of the rule. Check that the integration succeeds
on the generated polytopes ``tests/fixtures/synth_failures.json`` and
produces values which overlap their references (when these exist).
"""
import dataclasses
import itertools
import json
import os
import random

from sieve_integral_brion import (evaluate as config_evaluate, geometry as config_geometry,
                                  pieces as config_pieces, rules as config_rules)

CONFIG_FIELD = RealBallField(200)

# 1 <= t_1 <= 2, t_2, t_3 >= 1, t_2 + t_3 <= 4: an edge along e_2 - e_3, and
# t_2, t_3 have the same range [1, 3]
CONFIG_EDGE = Polyhedron(ieqs=[(-1, 1, 0, 0), (2, -1, 0, 0), (-1, 0, 1, 0), (-1, 0, 0, 1),
                               (4, 0, -1, -1)])

# t_j >= 1/5 on t_1 + t_2 + t_3 = 1: edges along every e_i - e_j, equal ranges
CONFIG_SIMPLEX = Polyhedron(eqns=[(-1, 1, 1, 1)],
                            ieqs=[(-1/5, 1, 0, 0), (-1/5, 0, 1, 0), (-1/5, 0, 0, 1)])
CONFIG_SEGMENT = Polyhedron(vertices=[(1/4, 3/4), (3/4, 1/4)])

# the grid which cuts the polytopes of the fixtures
SYNTH_FACTEUR = 5/2


def config_cases():
    r"""
    Return the polytopes with bad pairs, whether they lie on `\sum_j t_j = 1`,
    a precision at which their rule has an odd number of nodes, and their
    integrals.
    """
    C = ComplexBallField(CONFIG_FIELD.precision())
    edge = (CONFIG_FIELD(2).log()
            * C.integral(lambda t, a: (4 - t).log(analytic=a) / t, 1, 3).real())
    simplex = C.integral(lambda t, a: 2 * (4 - 5 * t).log(analytic=a) / (t * (1 - t)),
                         1/5, 3/5).real()
    return [("edge e_2 - e_3", CONFIG_EDGE, False, 16, edge),
            ("simplex", CONFIG_SIMPLEX, True, 14, simplex)]


def test_config_bad_pairs():
    edge = config_pieces.prepare_piece(CONFIG_EDGE, False, 16)
    assert config_geometry.bad_pairs(edge.generators) == {frozenset({1, 2})}, edge.generators
    # t_3 in [2, 4]: the edge along e_2 - e_3 is along (0, 2, -3) after rescaling
    shifted = Polyhedron(ieqs=[(-1, 1, 0, 0), (2, -1, 0, 0), (-1, 0, 1, 0), (-2, 0, 0, 1),
                               (5, 0, -1, -1)])
    piece = config_pieces.prepare_piece(shifted, False, 16)
    assert not config_geometry.bad_pairs(piece.generators), piece.generators
    simplex = config_pieces.prepare_piece(CONFIG_SIMPLEX, True, 14)
    assert len(config_geometry.bad_pairs(simplex.generators)) == 3, simplex.generators


def test_config_choose():
    rng = random.Random(int(8))
    for d in range(2, 10):
        k = d - 1
        pairs = [frozenset(p) for p in itertools.combinations(range(d), 2)]
        for m in range(1, 13):
            some = set(rng.sample(pairs, rng.randint(1, len(pairs))))
            for bad in (set(), some, set(pairs)):
                config = config_rules.choose_config(m, d, bad)
                assert sorted(config.order) == list(range(d)), (m, d, config)
                assert len(config.ms) == k and min(config.ms) >= m, (m, d, config)
                assert not any(config_rules.jacobi_rational_roots(n, k - 1 - i)
                               for i, n in enumerate(config.ms[:-1])), (m, d, config)
                assert not (config.ms[-1] % 2 and frozenset(config.order[-2:]) in bad), (
                    m, d, bad, config)
                if not bad:
                    assert config.order == tuple(range(d)), (m, d, config)
                assert config.ms[-1] == m or (config.ms[-1] == m + 1 and bad == set(pairs)), (
                    m, d, bad, config)
                for attempt in (1, 2):
                    other = config_rules.next_config(config, attempt, bad, rng)
                    assert sorted(other.order) == list(range(d)), (config, other)
                    assert all(a >= b for a, b in zip(other.ms, config.ms)), (config, other)
                    assert not (other.ms[-1] % 2 and frozenset(other.order[-2:]) in bad), (
                        bad, other)
    # the rational roots 1/15 and 1/3 of m = 2 for the weight (1 - u)^6
    assert config_rules.choose_config(2, 8).ms == (3, 2, 2, 2, 2, 2, 2)


def test_config_singular():
    counts = config_evaluate.counts
    for name, polytope, border, precision, expected in config_cases():
        piece = config_pieces.prepare_piece(polytope, border, precision)
        d = polytope.ambient_dimension()
        m = (piece.degree + 1) // 2
        assert m % 2, (name, m)
        # the identity order, with m nodes in every direction: the middle node
        # of the last direction lies on x_{d-1} = x_d
        identity = config_rules.RuleConfig(tuple(range(d)), (m,) * (d - 1))
        try:
            config_evaluate.rule_sum(piece, 128, identity)
        except config_evaluate.SingularNode as exc:
            assert exc.nodes and all(node[-1] == m // 2 for node in exc.nodes), (name, exc)
        else:
            raise AssertionError(f"{name}: no singular node with {identity}")
        field = RealBallField(precision)
        before = dict(counts)
        value = config_evaluate.integrate_piece(piece, field, polytope, border,
                                                config_pieces.PLAIN)
        check_overlap(value, expected, name)
        assert value.rad() < 2^(1 - precision), (name, value)
        assert (counts["singular"], counts["reconfigured"]) == (
            before["singular"], before["reconfigured"]), (name, piece.config)
        forced = dataclasses.replace(piece, config=identity)
        value = config_evaluate.integrate_piece(forced, field, polytope, border,
                                                config_pieces.PLAIN)
        check_overlap(value, expected, f"{name}, reconfigured")
        assert counts["reconfigured"] - before["reconfigured"] == 1, name


def test_config_reconfiguration():
    counts = config_evaluate.counts
    field = RealBallField(30)
    piece = config_pieces.prepare_piece(CONFIG_SEGMENT, True, 30)
    m = (piece.degree + 1) // 2
    odd = dataclasses.replace(
        piece, config=config_rules.RuleConfig((0, 1), (2 * (m // 2) + 1,)))
    expected = 2 * CONFIG_FIELD(3).log()
    saved = config_evaluate.next_config, config_evaluate.prepare_piece
    try:
        # every reconfiguration keeps the singular node, until the piece is
        # prepared again around perturbed centres
        config_evaluate.next_config = lambda config, attempt, bad, rng: config
        before = counts["reconfigured"]
        value = config_evaluate.integrate_piece(odd, field, CONFIG_SEGMENT, True,
                                                config_pieces.PLAIN)
        check_overlap(value, expected, "perturbed centres")
        assert counts["reconfigured"] - before == config_evaluate.ATTEMPTS - 1
        # and when that keeps it too
        config_evaluate.prepare_piece = lambda *arguments: odd
        try:
            config_evaluate.integrate_piece(odd, field, CONFIG_SEGMENT, True,
                                            config_pieces.PLAIN)
        except ArithmeticError as exc:
            assert type(exc) is ArithmeticError, exc
            assert "singular nodes under" in str(exc) and "(1, -1)" in str(exc), exc
        else:
            raise AssertionError("no ArithmeticError after the last configuration")
    finally:
        config_evaluate.next_config, config_evaluate.prepare_piece = saved


def synth_fixtures():
    r"""
    The entries of ``tests/fixtures/synth_failures.json``, each with its
    polytope on `\sum_j t_j = 1`.
    """
    with open(os.path.join(REPOSITORY, "tests", "fixtures", "synth_failures.json")) as f:
        entries = json.load(f)
    for entry in entries:
        d = len(entry["ieqs"][0]) - 1
        entry["polytope"] = Polyhedron(ieqs=entry["ieqs"], eqns=[[-1] + [1] * d],
                                       base_ring=QQ)
    return entries


def test_config_synth_failures():
    field = RealBallField(64)
    counts = config_evaluate.counts
    checked = 0
    for entry in synth_fixtures():
        if entry["library"] is None:
            continue
        singular = counts["singular"]
        value = field.zero()
        for part in config_geometry.grid_pieces(entry["polytope"], SYNTH_FACTEUR):
            piece = config_pieces.prepare_piece(part, True, entry["precision"])
            value += config_evaluate.integrate_piece(piece, field, part, True,
                                                     config_pieces.PLAIN)
        assert counts["singular"] == singular, (entry["id"], counts["singular"] - singular)
        check_overlap(value, reference_ball(entry["library"]), f"{entry['id']}, library")
        checked += 1
    assert checked == 4, checked


def test_config_synth_exit_6():
    # cut by the grid and prepared; the prototype configuration of the
    # pieces with a bad pair last and an odd number of nodes is singular
    entry = next(e for e in synth_fixtures() if e["library"] is None)
    singular = config_evaluate.counts["singular"]
    stuck = []
    for part in config_geometry.grid_pieces(entry["polytope"], SYNTH_FACTEUR):
        piece = config_pieces.prepare_piece(part, True, entry["precision"])
        if isinstance(piece, config_pieces.Trivial):
            continue
        d = len(piece.centres)
        m = (piece.degree + 1) // 2
        if m % 2 and frozenset((d - 2, d - 1)) in config_geometry.bad_pairs(piece.generators):
            stuck.append((m, part, piece))
    assert stuck, entry["id"]
    field = RealBallField(64)
    for m, part, piece in sorted(stuck, key=lambda s: s[0])[:3]:
        value = config_evaluate.integrate_piece(piece, field, part, True, config_pieces.PLAIN)
        assert value.rad() < 2^(1 - entry["precision"]), (part.vertices_list(), value)
    assert config_evaluate.counts["singular"] == singular


TESTS = {
    "config/bad-pairs": (test_config_bad_pairs, ("quick",)),
    "config/choose": (test_config_choose, ("quick",)),
    "config/singular": (test_config_singular, ("quick",)),
    "config/reconfiguration": (test_config_reconfiguration, ("quick",)),
    "config/synth-failures": (test_config_synth_failures, ("long",)),
    "config/synth-exit-6": (test_config_synth_exit_6, ("long",)),
}
