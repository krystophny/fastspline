"""
Setup script for FastSpline package.
"""
from setuptools import setup, find_packages, Extension
from pathlib import Path
import numpy as np

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Define the C extension
bispev_extension = Extension(
    'fastspline._bispev_c',
    sources=[
        'src/fortran/bispev.f',
        'src/fortran/fpbisp.f', 
        'src/fortran/fpbspl.f',
        'src/c/bispev_wrapper.c'
    ],
    include_dirs=['include', np.get_include()],
    extra_compile_args=['-O3'],
    extra_f77_compile_args=['-O3', '-fPIC'],
    extra_f90_compile_args=['-O3', '-fPIC'],
)

setup(
    name="fastspline",
    version="0.1.0",
    author="FastSpline Contributors",
    description="High-performance bivariate spline interpolation implementations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/krystophny/fastspline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.6.0",
    ],
    extras_require={
        "numba": ["numba>=0.55.0"],
        "dev": [
            "pytest>=6.0",
            "matplotlib>=3.3.0",
        ],
    },
    ext_modules=[bispev_extension],
    package_data={
        'fastspline': ['lib/*.so'],
    },
    include_package_data=True,
    zip_safe=False,
)