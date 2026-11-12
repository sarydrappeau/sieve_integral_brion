r"""
Import the compiled kernel, building it first if needed.

The kernel is built by ``sage -python setup.py build_ext --inplace``. When its
compiled module is missing, or older than ``_kernel.pyx``, :func:`load_kernel`
builds it in the package directory with :func:`sage.misc.cython.cython`, under
a file lock, unless the environment variable ``SIEVE_BRION_AUTOBUILD`` is
``0``.

AUTHORS:

- Sary Drappeau

AI DISCLOSURE:

Claude Opus 5.5 wrote this module.
"""
import fcntl
import glob
import importlib
import importlib.machinery
import importlib.util
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "_kernel.pyx")
NAME = "sieve_integral_brion._kernel"


def compiled_module():
    r"""
    Return the path of the compiled kernel in the package directory, or
    ``None``.
    """
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = os.path.join(HERE, "_kernel" + suffix)
        if os.path.exists(path):
            return path
    return None


def is_stale():
    r"""
    Whether the compiled kernel is missing or older than its source.
    """
    path = compiled_module()
    return path is None or os.path.getmtime(path) < os.path.getmtime(SOURCE)


def build():
    r"""
    Compile ``_kernel.pyx`` into the package directory with Sage's Cython
    tools, as ``_kernel`` followed by the extension suffix of this Python.
    """
    from sage.misc.cython import cython
    with open(os.path.join(HERE, ".build.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if not is_stale():
                return
            # cython() names the module after its file and builds it in the
            # current directory, from a copy made there
            work = os.path.join(HERE, "build", "autobuild")
            os.makedirs(work, exist_ok=True)
            here = os.getcwd()
            os.chdir(work)
            try:
                cython(SOURCE, create_local_so_file=True, compile_message=True,
                       sage_namespace=False)
            finally:
                os.chdir(here)
            built = [p for p in glob.glob(os.path.join(work, "_kernel*"))
                     if p.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))]
            if not built:
                raise ImportError(f"building {SOURCE} produced no module")
            suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
            shutil.move(built[0], os.path.join(HERE, "_kernel" + suffix))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def load_kernel():
    r"""
    Return the kernel module, building it first if it is stale and
    ``SIEVE_BRION_AUTOBUILD`` is not ``0``.
    """
    if is_stale():
        if os.environ.get("SIEVE_BRION_AUTOBUILD", "1") == "0":
            if compiled_module() is None:
                raise ImportError(
                    "the kernel of sieve_integral_brion is not built: run "
                    "'sage -python setup.py build_ext --inplace'")
        else:
            build()
    if NAME in sys.modules:
        return sys.modules[NAME]
    return importlib.import_module(NAME)
