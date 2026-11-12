r"""
Access to the delay-differential solutions of the ``sieve_integral`` library.

The Dickman and Buchstab functions come from its module
``number_theoretic_dde_solutions.py``, taken from the directory named by the
environment variable ``SIEVE_INTEGRAL_CODE`` if it is set. Otherwise it is
looked for in ``../sieve_integral/code`` next to this repository, and failing
that it is downloaded from the library's GitHub repository into
``$XDG_CACHE_HOME/sieve_integral_brion`` (by default
``~/.cache/sieve_integral_brion``), where later imports find it; delete it to
download again.

AUTHORS:

- Sary Drappeau

AI DISCLOSURE:

Claude Opus 5.5 wrote this module.
"""
import importlib.util
import os
import sys
import urllib.request
import warnings
from functools import lru_cache

MODULE = "number_theoretic_dde_solutions"
FILE = MODULE + ".py"
REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIRECTORY = os.path.normpath(
    os.path.join(REPOSITORY, os.pardir, "sieve_integral", "code"))
URL = ("https://raw.githubusercontent.com/sarydrappeau/sieve_integral/main/code/"
       + FILE)


def cache_directory():
    r"""
    Return the directory where the downloaded module is kept.

    EXAMPLES::

        sage: from sieve_integral_brion.dde import cache_directory
        sage: cache_directory().endswith("sieve_integral_brion")
        True
    """
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(root, "sieve_integral_brion")


def download(directory):
    r"""
    Download ``number_theoretic_dde_solutions.py`` from the library's GitHub
    repository into ``directory``, and return its path.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, FILE)
    with urllib.request.urlopen(URL, timeout=30) as response:
        source = response.read()
    with open(path + ".part", "wb") as f:
        f.write(source)
    os.replace(path + ".part", path)
    return path


def module_path():
    r"""
    Return the path of ``number_theoretic_dde_solutions.py``, downloading it
    if it is neither in ``SIEVE_INTEGRAL_CODE``, when set, nor in
    ``../sieve_integral/code`` or the cache.
    """
    if "SIEVE_INTEGRAL_CODE" in os.environ:
        path = os.path.join(os.environ["SIEVE_INTEGRAL_CODE"], FILE)
        if not os.path.isfile(path):
            raise ImportError(f"{FILE}, from the sieve_integral library, is not "
                              "in the directory named by SIEVE_INTEGRAL_CODE, "
                              f"{os.environ['SIEVE_INTEGRAL_CODE']}")
        return path
    for directory in (DEFAULT_DIRECTORY, cache_directory()):
        path = os.path.join(directory, FILE)
        if os.path.isfile(path):
            return path
    warnings.warn(f"downloading {URL} into {cache_directory()}")
    try:
        return download(cache_directory())
    except OSError as exc:
        raise ImportError(f"{FILE}, from the sieve_integral library, is not in "
                          f"{DEFAULT_DIRECTORY}, and downloading it from {URL} "
                          f"failed ({exc}): set the environment variable "
                          "SIEVE_INTEGRAL_CODE to the directory that holds it") from exc


@lru_cache(maxsize=None)
def dde_module():
    r"""
    Return the module ``number_theoretic_dde_solutions``, importing it from
    :func:`module_path` unless it is already imported.

    EXAMPLES::

        sage: from sieve_integral_brion.dde import dde_module
        sage: dde_module().BuchstabB(precision=53)(3).overlaps(RBF(1 + log(2)))
        True
    """
    if MODULE in sys.modules:
        return sys.modules[MODULE]
    spec = importlib.util.spec_from_file_location(MODULE, module_path())
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[MODULE]
        raise
    return module


def BuchstabB(**kwds):
    r"""
    Return ``BuchstabB(**kwds)`` from ``number_theoretic_dde_solutions``.
    """
    return dde_module().BuchstabB(**kwds)


def DickmanRho(**kwds):
    r"""
    Return ``DickmanRho(**kwds)`` from ``number_theoretic_dde_solutions``.
    """
    return dde_module().DickmanRho(**kwds)
