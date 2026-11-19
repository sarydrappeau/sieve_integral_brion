r"""
Build the Cython module ``_brion_sum`` of ``sieve_integral_brion`` in place::

    sage -python setup.py build_ext --inplace

Importing the package builds it the same way when the compiled module is
missing or older than its source (see ``sieve_integral_brion/_brion_sum_loader.py``).
"""
from setuptools import Extension, setup
from Cython.Build import cythonize
from sage.env import sage_include_directories

extensions = [Extension("sieve_integral_brion._brion_sum",
                        ["sieve_integral_brion/_brion_sum.pyx"],
                        include_dirs=sage_include_directories())]

setup(name="sieve_integral_brion",
      packages=["sieve_integral_brion"],
      ext_modules=cythonize(extensions,
                            include_path=sage_include_directories(),
                            compiler_directives={"language_level": 3,
                                                 "boundscheck": False,
                                                 "wraparound": False,
                                                 "cdivision": True}))
