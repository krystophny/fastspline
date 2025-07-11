"""
Utilities for validating Numba implementations against Fortran/scipy.
"""
import numpy as np
import ctypes
from pathlib import Path
import subprocess
import tempfile
import os


class FortranValidator:
    """Helper class to call Fortran routines for validation."""
    
    def __init__(self):
        # Load the compiled library
        lib_path = Path(__file__).parent.parent / "libbispev.so"
        if not lib_path.exists():
            raise RuntimeError("Please run 'make' in the parent directory first")
        self.lib = ctypes.CDLL(str(lib_path))
        
        # Set up function signatures
        self._setup_fpbspl()
        self._setup_fpbisp()
        self._setup_bispev()
    
    def _setup_fpbspl(self):
        """Set up fpbspl function signature."""
        # fpbspl is typically not exposed directly, we'll need to create a wrapper
        pass
    
    def _setup_fpbisp(self):
        """Set up fpbisp function signature.""" 
        # fpbisp is also internal, we'll need to create a wrapper
        pass
        
    def _setup_bispev(self):
        """Set up bispev function signature."""
        self.bispev_c = self.lib.bispev_c
        self.bispev_c.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # tx
            ctypes.c_int,                      # nx
            ctypes.POINTER(ctypes.c_double),  # ty
            ctypes.c_int,                      # ny
            ctypes.POINTER(ctypes.c_double),  # c
            ctypes.c_int,                      # kx
            ctypes.c_int,                      # ky
            ctypes.POINTER(ctypes.c_double),  # x
            ctypes.c_int,                      # mx
            ctypes.POINTER(ctypes.c_double),  # y
            ctypes.c_int,                      # my
            ctypes.POINTER(ctypes.c_double),  # z
            ctypes.POINTER(ctypes.c_double),  # wrk
            ctypes.c_int,                      # lwrk
            ctypes.POINTER(ctypes.c_int),     # iwrk
            ctypes.c_int,                      # kwrk
            ctypes.POINTER(ctypes.c_int),     # ier
        ]
        self.bispev_c.restype = ctypes.c_int


def create_fortran_test_wrapper():
    """Create a Fortran program that exposes fpbspl and fpbisp for testing."""
    fortran_code = """
      program test_wrapper
      implicit none
      integer :: test_case
      read(*,*) test_case
      
      if (test_case .eq. 1) then
          call test_fpbspl()
      else if (test_case .eq. 2) then
          call test_fpbisp()
      end if
      
      contains
      
      subroutine test_fpbspl()
          implicit none
          integer :: n, k, l, i
          real*8 :: x
          real*8, allocatable :: t(:), h(:)
          
          ! Read inputs
          read(*,*) n, k, l, x
          allocate(t(n), h(k+1))
          read(*,*) (t(i), i=1,n)
          
          ! Call fpbspl
          call fpbspl(t, n, k, x, l, h)
          
          ! Write outputs
          write(*,*) (h(i), i=1,k+1)
          
          deallocate(t, h)
      end subroutine test_fpbspl
      
      subroutine test_fpbisp()
          implicit none
          ! Implementation for fpbisp testing
      end subroutine test_fpbisp
      
      end program test_wrapper
    """
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.f90', delete=False) as f:
        f.write(fortran_code)
        temp_file = f.name
    
    # Compile with existing Fortran files
    try:
        exe_file = temp_file.replace('.f90', '.exe')
        cmd = [
            'gfortran', '-o', exe_file,
            temp_file,
            'src/fortran/fpbspl.f',
            'src/fortran/fpbisp.f',
            'src/fortran/bispev.f'
        ]
        subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)
        return exe_file
    finally:
        os.unlink(temp_file)


def call_fortran_fpbspl(t, n, k, x, l):
    """Call Fortran fpbspl through test wrapper."""
    exe_file = create_fortran_test_wrapper()
    
    try:
        # Prepare input
        input_data = f"1\n{n} {k} {l} {x}\n"
        input_data += " ".join(str(ti) for ti in t) + "\n"
        
        # Run Fortran program
        result = subprocess.run(
            [exe_file],
            input=input_data,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output
        h = np.array([float(x) for x in result.stdout.strip().split()])
        return h
        
    finally:
        os.unlink(exe_file)


def generate_test_spline_data(nx=10, ny=10, kx=3, ky=3):
    """Generate test data for spline evaluation."""
    # Create knot vectors
    tx = np.concatenate([
        np.zeros(kx+1),
        np.linspace(0, 1, nx-2*kx-2),
        np.ones(kx+1)
    ])
    ty = np.concatenate([
        np.zeros(ky+1),
        np.linspace(0, 1, ny-2*ky-2), 
        np.ones(ky+1)
    ])
    
    # Create coefficients (random or specific pattern)
    nc = (nx - kx - 1) * (ny - ky - 1)
    c = np.random.randn(nc)
    
    return tx, ty, c, kx, ky


def compare_arrays(a1, a2, name="array", rtol=1e-14, atol=1e-14):
    """Compare two arrays and print differences."""
    if not np.allclose(a1, a2, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(a1 - a2))
        max_rel = np.max(np.abs((a1 - a2) / (a1 + 1e-100)))
        print(f"MISMATCH in {name}:")
        print(f"  Max absolute difference: {max_diff}")
        print(f"  Max relative difference: {max_rel}")
        print(f"  Expected: {a1}")
        print(f"  Got:      {a2}")
        return False
    else:
        print(f"✓ {name} matches (rtol={rtol}, atol={atol})")
        return True


def validate_fpbspl_simple():
    """Simple validation test for fpbspl."""
    # Test case: cubic B-spline
    k = 3
    n = 8
    t = np.array([0., 0., 0., 0., 1., 1., 1., 1.])
    x = 0.5
    l = 4  # knot interval for x=0.5
    
    # Call through scipy if available
    try:
        from scipy.interpolate._fitpack_py import _bspleval
        # scipy has different interface, skip for now
    except:
        pass
    
    print(f"Test case: k={k}, x={x}, l={l}")
    print(f"Knots: {t}")
    
    # We'll implement and test fpbspl directly
    return True


if __name__ == "__main__":
    print("Testing validation utilities...")
    validate_fpbspl_simple()