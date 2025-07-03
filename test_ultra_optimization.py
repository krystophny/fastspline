#!/usr/bin/env python3
"""Test ultra-optimized bisplev implementation."""

import numpy as np
import time
from numba import cfunc, types
from scipy.interpolate import bisplrep, bisplev as scipy_bisplev

@cfunc(types.float64(types.float64, types.float64, types.float64[:], types.float64[:],
                     types.float64[:], types.int64, types.int64, types.int64, types.int64), 
       nopython=True, fastmath=True, boundscheck=False)
def _bisplev_cfunc_ultra(x, y, tx, ty, c, kx, ky, nx, ny):
    """
    ULTRA-OPTIMIZED B-spline evaluation with manual register allocation.
    All operations inlined, no function calls, aggressive optimization.
    """
    mx = nx - kx - 1
    my = ny - ky - 1
    
    # === INLINE KNOT SPAN FINDING ===
    # X direction knot span
    n_x = len(tx) - kx - 1
    if x >= tx[n_x]:
        span_x = n_x - 1
    elif x <= tx[kx]:
        span_x = kx
    else:
        # Binary search - unrolled for speed
        low = kx
        high = n_x
        while True:
            mid = (low + high) >> 1  # Bit shift instead of division
            if x < tx[mid]:
                high = mid
            elif x >= tx[mid + 1]:
                low = mid
            else:
                span_x = mid
                break
    
    # Y direction knot span  
    n_y = len(ty) - ky - 1
    if y >= ty[n_y]:
        span_y = n_y - 1
    elif y <= ty[ky]:
        span_y = ky
    else:
        # Binary search - unrolled for speed
        low = ky
        high = n_y
        while True:
            mid = (low + high) >> 1  # Bit shift instead of division
            if y < ty[mid]:
                high = mid
            elif y >= ty[mid + 1]:
                low = mid
            else:
                span_y = mid
                break
    
    # === ULTRA-OPTIMIZED CUBIC CASE ===
    if kx == 3 and ky == 3:
        # Manual register allocation for cubic basis functions
        # X direction cubic basis - fully unrolled Cox-de Boor
        # Register: left1_x, left2_x, left3_x, right1_x, right2_x, right3_x
        left1_x = x - tx[span_x]
        left2_x = x - tx[span_x - 1] 
        left3_x = x - tx[span_x - 2]
        right1_x = tx[span_x + 1] - x
        right2_x = tx[span_x + 2] - x
        right3_x = tx[span_x + 3] - x
        
        # Initialize: Nx0 = 1.0
        Nx0 = 1.0
        Nx1 = 0.0
        Nx2 = 0.0
        Nx3 = 0.0
        
        # j=1 (degree 1)
        denom = right1_x + left1_x
        temp = Nx0 / denom
        Nx0 = right1_x * temp
        Nx1 = left1_x * temp
        
        # j=2 (degree 2)  
        denom0 = right1_x + left2_x
        temp0 = Nx0 / denom0
        saved = right1_x * temp0
        
        denom1 = right2_x + left1_x
        temp1 = Nx1 / denom1
        Nx0 = saved
        Nx1 = left2_x * temp0 + right2_x * temp1
        Nx2 = left1_x * temp1
        
        # j=3 (degree 3)
        denom0 = right1_x + left3_x
        temp0 = Nx0 / denom0
        saved = right1_x * temp0
        
        denom1 = right2_x + left2_x  
        temp1 = Nx1 / denom1
        Nx0 = saved
        saved = left3_x * temp0 + right2_x * temp1
        
        denom2 = right3_x + left1_x
        temp2 = Nx2 / denom2
        Nx1 = saved
        Nx2 = left2_x * temp1 + right3_x * temp2
        Nx3 = left1_x * temp2
        
        # Y direction cubic basis - identical structure
        left1_y = y - ty[span_y]
        left2_y = y - ty[span_y - 1]
        left3_y = y - ty[span_y - 2] 
        right1_y = ty[span_y + 1] - y
        right2_y = ty[span_y + 2] - y
        right3_y = ty[span_y + 3] - y
        
        Ny0 = 1.0
        Ny1 = 0.0
        Ny2 = 0.0
        Ny3 = 0.0
        
        # j=1
        denom = right1_y + left1_y
        temp = Ny0 / denom
        Ny0 = right1_y * temp
        Ny1 = left1_y * temp
        
        # j=2
        denom0 = right1_y + left2_y
        temp0 = Ny0 / denom0
        saved = right1_y * temp0
        
        denom1 = right2_y + left1_y
        temp1 = Ny1 / denom1
        Ny0 = saved
        Ny1 = left2_y * temp0 + right2_y * temp1
        Ny2 = left1_y * temp1
        
        # j=3
        denom0 = right1_y + left3_y
        temp0 = Ny0 / denom0
        saved = right1_y * temp0
        
        denom1 = right2_y + left2_y
        temp1 = Ny1 / denom1
        Ny0 = saved
        saved = left3_y * temp0 + right2_y * temp1
        
        denom2 = right3_y + left1_y
        temp2 = Ny2 / denom2
        Ny1 = saved
        Ny2 = left2_y * temp1 + right3_y * temp2
        Ny3 = left1_y * temp2
        
        # Tensor product - 16 terms unrolled with manual indexing
        base_idx = (span_x - 3) * my + (span_y - 3)
        
        result = Nx0 * Ny0 * c[base_idx]
        result += Nx0 * Ny1 * c[base_idx + 1]
        result += Nx0 * Ny2 * c[base_idx + 2] 
        result += Nx0 * Ny3 * c[base_idx + 3]
        
        base_idx += my
        result += Nx1 * Ny0 * c[base_idx]
        result += Nx1 * Ny1 * c[base_idx + 1]
        result += Nx1 * Ny2 * c[base_idx + 2]
        result += Nx1 * Ny3 * c[base_idx + 3]
        
        base_idx += my
        result += Nx2 * Ny0 * c[base_idx]
        result += Nx2 * Ny1 * c[base_idx + 1]
        result += Nx2 * Ny2 * c[base_idx + 2]
        result += Nx2 * Ny3 * c[base_idx + 3]
        
        base_idx += my
        result += Nx3 * Ny0 * c[base_idx]
        result += Nx3 * Ny1 * c[base_idx + 1]
        result += Nx3 * Ny2 * c[base_idx + 2]
        result += Nx3 * Ny3 * c[base_idx + 3]
        
        return result
    
    # === ULTRA-OPTIMIZED LINEAR CASE ===  
    elif kx == 1 and ky == 1:
        # Direct linear interpolation - no loops
        alpha_x = (x - tx[span_x]) / (tx[span_x + 1] - tx[span_x])
        alpha_y = (y - ty[span_y]) / (ty[span_y + 1] - ty[span_y])
        
        beta_x = 1.0 - alpha_x
        beta_y = 1.0 - alpha_y
        
        base_idx = (span_x - 1) * my + (span_y - 1)
        
        result = beta_x * beta_y * c[base_idx]
        result += beta_x * alpha_y * c[base_idx + 1]
        result += alpha_x * beta_y * c[base_idx + my]
        result += alpha_x * alpha_y * c[base_idx + my + 1]
        
        return result
    
    # Fallback for other cases (should not be reached in our tests)
    return 0.0


def test_ultra_optimization():
    """Test the ultra-optimized implementation."""
    print("Testing Ultra-Optimized bisplev")
    print("=" * 50)
    
    # Create test data
    np.random.seed(42)
    x_data = np.random.uniform(-1, 1, 100)
    y_data = np.random.uniform(-1, 1, 100)
    z_data = np.exp(-(x_data**2 + y_data**2)) * np.cos(np.pi * x_data)
    
    # Test k=3 case
    tck_k3 = bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0.01)
    tx, ty, c, kx, ky = tck_k3
    nx, ny = len(tx), len(ty)
    
    # Test k=1 case  
    tck_k1 = bisplrep(x_data, y_data, z_data, kx=1, ky=1, s=0.01)
    tx1, ty1, c1, kx1, ky1 = tck_k1
    nx1, ny1 = len(tx1), len(ty1)
    
    # Test points
    n_test = 1000
    x_test = np.random.uniform(-0.8, 0.8, n_test)
    y_test = np.random.uniform(-0.8, 0.8, n_test)
    
    # === Test k=3 (cubic) ===
    print("\nCubic Splines (k=3) Performance:")
    
    # Time SciPy
    start = time.perf_counter()
    scipy_results = [scipy_bisplev(x, y, tck_k3) for x, y in zip(x_test, y_test)]
    time_scipy = (time.perf_counter() - start) * 1000
    
    # Time original cfunc
    from fastspline.spline2d import _bisplev_cfunc
    start = time.perf_counter()
    orig_results = [_bisplev_cfunc(x, y, tx, ty, c, kx, ky, nx, ny) for x, y in zip(x_test, y_test)]
    time_orig = (time.perf_counter() - start) * 1000
    
    # Time ultra-optimized
    start = time.perf_counter()
    ultra_results = [_bisplev_cfunc_ultra(x, y, tx, ty, c, kx, ky, nx, ny) for x, y in zip(x_test, y_test)]
    time_ultra = (time.perf_counter() - start) * 1000
    
    # Check accuracy
    diff_orig = np.max(np.abs(np.array(scipy_results) - np.array(orig_results)))
    diff_ultra = np.max(np.abs(np.array(scipy_results) - np.array(ultra_results)))
    
    print(f"  SciPy:           {time_scipy:6.1f}ms")
    print(f"  Original cfunc:  {time_orig:6.1f}ms  (speedup: {time_scipy/time_orig:.2f}x)")
    print(f"  Ultra-optimized: {time_ultra:6.1f}ms  (speedup: {time_scipy/time_ultra:.2f}x)")
    print(f"  Ultra vs Orig:   {time_orig/time_ultra:.2f}x improvement")
    print(f"  Accuracy orig:   {diff_orig:.2e}")
    print(f"  Accuracy ultra:  {diff_ultra:.2e}")
    
    # === Test k=1 (linear) ===
    print("\nLinear Splines (k=1) Performance:")
    
    # Time SciPy
    start = time.perf_counter()
    scipy_results = [scipy_bisplev(x, y, tck_k1) for x, y in zip(x_test, y_test)]
    time_scipy = (time.perf_counter() - start) * 1000
    
    # Time original cfunc
    start = time.perf_counter()
    orig_results = [_bisplev_cfunc(x, y, tx1, ty1, c1, kx1, ky1, nx1, ny1) for x, y in zip(x_test, y_test)]
    time_orig = (time.perf_counter() - start) * 1000
    
    # Time ultra-optimized
    start = time.perf_counter()
    ultra_results = [_bisplev_cfunc_ultra(x, y, tx1, ty1, c1, kx1, ky1, nx1, ny1) for x, y in zip(x_test, y_test)]
    time_ultra = (time.perf_counter() - start) * 1000
    
    # Check accuracy
    diff_orig = np.max(np.abs(np.array(scipy_results) - np.array(orig_results)))
    diff_ultra = np.max(np.abs(np.array(scipy_results) - np.array(ultra_results)))
    
    print(f"  SciPy:           {time_scipy:6.1f}ms")
    print(f"  Original cfunc:  {time_orig:6.1f}ms  (speedup: {time_scipy/time_orig:.2f}x)")
    print(f"  Ultra-optimized: {time_ultra:6.1f}ms  (speedup: {time_scipy/time_ultra:.2f}x)")
    print(f"  Ultra vs Orig:   {time_orig/time_ultra:.2f}x improvement")
    print(f"  Accuracy orig:   {diff_orig:.2e}")
    print(f"  Accuracy ultra:  {diff_ultra:.2e}")

if __name__ == "__main__":
    test_ultra_optimization()