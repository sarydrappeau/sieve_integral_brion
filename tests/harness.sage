r"""
Shared helpers of the test suite, loaded by ``tests/run_tests.sage`` from the
repository root.

The material of the ``sieve_integral`` library used by the tests (its examples,
reference values and h computation) is looked for in the directory named by
``SIEVE_INTEGRAL_ROOT``, by default ``../sieve_integral``. Unless
``SIEVE_INTEGRAL_CODE`` is set, the DDE module is taken from the ``code/``
directory of that root when it is there, and otherwise found as the package
finds it (:mod:`sieve_integral_brion.dde`).
"""
import json
import os
import sys
from contextlib import contextmanager
from unittest import SkipTest

REPOSITORY = os.getcwd()
if not os.path.isdir(os.path.join(REPOSITORY, "sieve_integral_brion")):
    raise RuntimeError("run the test suite from the repository root: "
                       f"{REPOSITORY} has no sieve_integral_brion/")
if REPOSITORY not in sys.path:
    sys.path.insert(0, REPOSITORY)

LIBRARY_ROOT = os.path.abspath(os.environ.get(
    "SIEVE_INTEGRAL_ROOT", os.path.join(REPOSITORY, os.pardir, "sieve_integral")))
if ("SIEVE_INTEGRAL_CODE" not in os.environ and os.path.isfile(
        os.path.join(LIBRARY_ROOT, "code", "number_theoretic_dde_solutions.py"))):
    os.environ["SIEVE_INTEGRAL_CODE"] = os.path.join(LIBRARY_ROOT, "code")


def library_path(*parts):
    r"""
    Return the path of ``parts`` under the library's root, raising
    ``SkipTest`` if it does not exist.
    """
    path = os.path.join(LIBRARY_ROOT, *parts)
    if not os.path.exists(path):
        raise SkipTest(f"{path} not found (set SIEVE_INTEGRAL_ROOT)")
    return path


def library_module():
    r"""
    Return the library's module ``sieve_integral``, imported from its
    ``code/`` directory, raising ``SkipTest`` if it cannot be.
    """
    code = library_path("code")
    if code not in sys.path:
        sys.path.append(code)
    try:
        import sieve_integral
    except ImportError as exc:
        raise SkipTest(f"the sieve_integral library does not import: {exc}")
    return sieve_integral


@contextmanager
def environment(**values):
    r"""
    Run the body with the environment variables ``values`` set, ``None``
    meaning unset.
    """
    saved = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def reference_ball(entry):
    r"""
    The ball of an entry ``{"mid", "rad", "prec"}`` of a JSON reference file.
    """
    return RealBallField(int(entry["prec"]))(QQ(entry["mid"])).add_error(QQ(entry["rad"]))


def check_overlap(value, expected, what):
    r"""
    Raise ``AssertionError`` unless ``value`` overlaps ``expected``.
    """
    if not value.overlaps(expected):
        raise AssertionError(f"{what}: {value} does not overlap {expected}")
