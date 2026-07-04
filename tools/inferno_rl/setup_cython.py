"""
Build script for Cython-optimized simulator modules.

Usage:
    cd tools/inferno_rl
    python setup_cython.py build_ext --inplace
    python -m tools.inferno_rl.verify_cython_backend --require-compiled

Produces:
    simulator/geometry.cp312-win_amd64.pyd
    simulator/pathfinding.cp312-win_amd64.pyd
    simulator/forecast_fast.cp312-win_amd64.pyd

The smoke check must import through tools.inferno_rl.* so it verifies the same
package path used by training.
"""
import os
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

# Ensure we build from the correct directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

numpy_include = np.get_include()

extensions = [
    Extension(
        "tools.inferno_rl.simulator.geometry",
        ["simulator/geometry.pyx"],
    ),
    Extension(
        "tools.inferno_rl.simulator.pathfinding",
        ["simulator/pathfinding.pyx"],
    ),
    Extension(
        "tools.inferno_rl.simulator.forecast_fast",
        ["simulator/forecast_fast.pyx"],
        include_dirs=[numpy_include],
    ),
]

setup(
    package_dir={
        "tools.inferno_rl": ".",
        "tools.inferno_rl.simulator": "simulator",
    },
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    ),
)
