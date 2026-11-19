r"""
The package imports, finds the DDE module, and builds its compiled module in
two possible ways (through setup.py or on first import).
"""
import shutil
import subprocess
import tempfile


def test_import():
    import sieve_integral_brion
    assert sieve_integral_brion.brion_sum.BrionSum.__name__ == "BrionSum"


def test_dde_import():
    from sieve_integral_brion.dde import dde_module
    value = dde_module().BuchstabB(precision=53)(3)
    check_overlap(value, RealBallField(100)(1 + log(2)), "B(3)")


def test_dde_missing():
    from sieve_integral_brion import dde
    saved = sys.modules.pop(dde.MODULE, None)
    dde.dde_module.cache_clear()
    try:
        with environment(SIEVE_INTEGRAL_CODE="/nonexistent/directory"):
            try:
                dde.dde_module()
            except ImportError as exc:
                assert "SIEVE_INTEGRAL_CODE" in str(exc), exc
            else:
                raise AssertionError("no ImportError for a missing module")
    finally:
        dde.dde_module.cache_clear()
        if saved is not None:
            sys.modules[dde.MODULE] = saved


def test_dde_download():
    import warnings
    from sieve_integral_brion import dde
    saved_module = sys.modules.pop(dde.MODULE, None)
    saved_default = dde.DEFAULT_DIRECTORY
    dde.dde_module.cache_clear()
    try:
        with tempfile.TemporaryDirectory() as cache:
            dde.DEFAULT_DIRECTORY = os.path.join(cache, "missing")
            with environment(SIEVE_INTEGRAL_CODE=None, XDG_CACHE_HOME=cache):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        module = dde.dde_module()
                except ImportError as exc:
                    if isinstance(exc.__cause__, OSError):
                        raise SkipTest(f"no download from {dde.URL}: {exc.__cause__}")
                    raise
                assert hasattr(module, "BuchstabB"), "the download has no BuchstabB"
                path = os.path.join(dde.cache_directory(), dde.FILE)
                assert os.path.isfile(path), f"{path} was not written"
    finally:
        dde.DEFAULT_DIRECTORY = saved_default
        dde.dde_module.cache_clear()
        sys.modules.pop(dde.MODULE, None)
        if saved_module is not None:
            sys.modules[dde.MODULE] = saved_module


def copy_sources(target):
    r"""
    Copy ``setup.py`` and the sources of the package, without any compiled
    module, into ``target``.
    """
    shutil.copy(os.path.join(REPOSITORY, "setup.py"), target)
    package = os.path.join(target, "sieve_integral_brion")
    os.makedirs(package)
    for name in os.listdir(os.path.join(REPOSITORY, "sieve_integral_brion")):
        if name.endswith((".py", ".pyx")):
            shutil.copy(os.path.join(REPOSITORY, "sieve_integral_brion", name), package)


CHECK_IMPORT = "import sieve_integral_brion as s; print(s.brion_sum.BrionSum.__name__)"


def sage_python(arguments, cwd, **env):
    r"""
    Run ``sage -python`` with ``arguments`` in ``cwd``, and return the
    completed process.
    """
    environ = dict(os.environ, **env)
    return subprocess.run(["sage", "-python"] + arguments, cwd=cwd, env=environ,
                          capture_output=True, text=True)


def test_build_autobuild():
    with tempfile.TemporaryDirectory() as target:
        copy_sources(target)
        disabled = sage_python(["-c", CHECK_IMPORT], target, SIEVE_BRION_AUTOBUILD="0")
        assert disabled.returncode != 0 and "not built" in disabled.stderr, disabled.stderr
        built = sage_python(["-c", CHECK_IMPORT], target)
        assert built.returncode == 0 and "BrionSum" in built.stdout, built.stderr


def test_build_setup():
    with tempfile.TemporaryDirectory() as target:
        copy_sources(target)
        build = sage_python(["setup.py", "build_ext", "--inplace"], target)
        assert build.returncode == 0, build.stderr[-2000:]
        loaded = sage_python(["-c", CHECK_IMPORT], target, SIEVE_BRION_AUTOBUILD="0")
        assert loaded.returncode == 0 and "BrionSum" in loaded.stdout, loaded.stderr


TESTS = {
    "scaffold/import": (test_import, ("quick",)),
    "scaffold/dde/import": (test_dde_import, ("quick",)),
    "scaffold/dde/missing": (test_dde_missing, ("quick",)),
    "scaffold/dde/download": (test_dde_download, ("long",)),
    "scaffold/build/autobuild": (test_build_autobuild, ("long",)),
    "scaffold/build/setup": (test_build_setup, ("long",)),
}
