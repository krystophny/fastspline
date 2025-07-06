import numpy as np
from numpy.distutils.core import setup, Extension

# Define the extension module
ext = Extension(
    name='dierckx_f2py',
    sources=[
        'thirdparty/dierckx/surfit.f',
        'thirdparty/dierckx/fpsurf.f',
        'thirdparty/dierckx/fpback.f',
        'thirdparty/dierckx/fpdisc.f',
        'thirdparty/dierckx/fpgivs.f',
        'thirdparty/dierckx/fprank.f',
        'thirdparty/dierckx/fprati.f',
        'thirdparty/dierckx/fprota.f',
        'thirdparty/dierckx/fporde.f',
        'thirdparty/dierckx/fpbspl.f',
        'dierckx_f2py.pyf'
    ],
    extra_f77_compile_args=['-fPIC'],
    libraries=['lapack', 'blas']
)

setup(
    name='dierckx_f2py',
    ext_modules=[ext]
)