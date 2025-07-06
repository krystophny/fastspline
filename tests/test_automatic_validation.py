#!/usr/bin/env python3
"""
AUTOMATIC VALIDATION TEST for bisplrep/bisplev cfunc
This test MUST always pass in the future - it validates core functionality
"""

import numpy as np
from scipy.interpolate import bisplrep, bisplev
import sys
sys.path.insert(0, '..')

from dierckx_cfunc import bisplrep_cfunc, bisplev_cfunc

def test_exact_match_requirement():
    """
    CRITICAL TEST: These cases MUST match SciPy exactly (< 1e-10 error)
    If this fails, the implementation is broken and must be fixed immediately
    """
    print("AUTOMATIC VALIDATION: EXACT MATCH REQUIREMENTS")
    print("=" * 80)
    
    test_cases = [
        # Simple cases that MUST work exactly
        {
            'name': 'Linear 2x2 grid',
            'x': np.array([0., 1., 0., 1.]),
            'y': np.array([0., 0., 1., 1.]),
            'z': np.array([1., 2., 3., 4.]),
            'kx': 1, 'ky': 1,
            'eval_points': [(0.5, 0.5), (0.0, 0.0), (1.0, 1.0)]
        },
        {
            'name': 'Quadratic 3x3 grid',
            'x': np.array([0., 1., 2., 0., 1., 2., 0., 1., 2.]),
            'y': np.array([0., 0., 0., 1., 1., 1., 2., 2., 2.]),
            'z': np.array([0., 1., 4., 1., 2., 5., 4., 5., 8.]),  # z = x^2 + y^2
            'kx': 2, 'ky': 2,
            'eval_points': [(0.5, 0.5), (1.0, 1.0), (1.5, 1.5)]
        },
        {
            'name': 'Constant surface',
            'x': np.array([0., 1., 0., 1.]),
            'y': np.array([0., 0., 1., 1.]),
            'z': np.array([5., 5., 5., 5.]),
            'kx': 1, 'ky': 1,
            'eval_points': [(0.5, 0.5), (0.25, 0.75)]
        }
    ]
    
    all_passed = True
    
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        
        try:
            # SciPy reference
            tck_scipy = bisplrep(case['x'], case['y'], case['z'], 
                               kx=case['kx'], ky=case['ky'], s=0)
            
            # Our implementation  
            tx, ty, c, kx, ky = bisplrep_cfunc(case['x'], case['y'], case['z'],
                                             kx=case['kx'], ky=case['ky'], s=0.0)
            
            # Test knot vectors match
            tx_error = np.max(np.abs(tx - tck_scipy[0]))
            ty_error = np.max(np.abs(ty - tck_scipy[1]))
            c_error = np.max(np.abs(c - tck_scipy[2]))
            
            knots_ok = tx_error < 1e-10 and ty_error < 1e-10 and c_error < 1e-10
            
            # Test evaluation at multiple points
            eval_ok = True
            max_eval_error = 0.0
            
            for x_pt, y_pt in case['eval_points']:
                z_scipy = bisplev(x_pt, y_pt, tck_scipy)
                z_cfunc = bisplev_cfunc(np.array([x_pt]), np.array([y_pt]), 
                                       tx, ty, c, kx, ky)[0, 0]
                
                error = abs(z_scipy - z_cfunc)
                max_eval_error = max(max_eval_error, error)
                if error >= 1e-10:
                    eval_ok = False
            
            if knots_ok and eval_ok:
                print(f"  ✓ PASS (max eval error: {max_eval_error:.2e})")
            else:
                print(f"  ✗ FAIL")
                print(f"    Knot errors: tx={tx_error:.2e}, ty={ty_error:.2e}, c={c_error:.2e}")
                print(f"    Max eval error: {max_eval_error:.2e}")
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ FAIL: Exception: {str(e)}")
            all_passed = False
    
    return all_passed

def test_boundary_conditions():
    """Test evaluation at domain boundaries"""
    print("\n" + "=" * 80)
    print("BOUNDARY CONDITIONS TEST")
    print("=" * 80)
    
    # Create 4x4 grid
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    X, Y = np.meshgrid(x, y)
    Z = X + Y  # Simple plane
    
    tck_scipy = bisplrep(X.flatten(), Y.flatten(), Z.flatten(), s=0)
    tx, ty, c, kx, ky = bisplrep_cfunc(X.flatten(), Y.flatten(), Z.flatten(), s=0.0)
    
    # Test at boundaries
    boundary_points = [
        (-1.0, -1.0, "Bottom-left corner"),
        (-1.0, 1.0, "Top-left corner"),
        (1.0, -1.0, "Bottom-right corner"),  
        (1.0, 1.0, "Top-right corner"),
        (0.0, -1.0, "Bottom edge"),
        (0.0, 1.0, "Top edge"),
        (-1.0, 0.0, "Left edge"),
        (1.0, 0.0, "Right edge")
    ]
    
    max_boundary_error = 0.0
    boundary_ok = True
    
    for x_pt, y_pt, desc in boundary_points:
        z_scipy = bisplev(x_pt, y_pt, tck_scipy)
        z_cfunc = bisplev_cfunc(np.array([x_pt]), np.array([y_pt]), 
                               tx, ty, c, kx, ky)[0, 0]
        
        error = abs(z_scipy - z_cfunc)
        max_boundary_error = max(max_boundary_error, error)
        
        if error < 1e-10:
            print(f"  ✓ {desc}: error={error:.2e}")
        else:
            print(f"  ✗ {desc}: error={error:.2e} (TOO LARGE)")
            boundary_ok = False
    
    return boundary_ok

def test_performance_requirement():
    """Test that our implementation isn't catastrophically slow"""
    print("\n" + "=" * 80)
    print("PERFORMANCE SANITY CHECK")
    print("=" * 80)
    
    import time
    
    # Medium sized problem
    n = 50
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    z = x**2 + y**2
    
    # Time SciPy
    start = time.perf_counter()
    tck_scipy = bisplrep(x, y, z, s=0)
    scipy_time = time.perf_counter() - start
    
    # Time our implementation
    start = time.perf_counter()
    tx, ty, c, kx, ky = bisplrep_cfunc(x, y, z, s=0.0)
    cfunc_time = time.perf_counter() - start
    
    slowdown = cfunc_time / scipy_time
    
    print(f"SciPy time: {scipy_time*1000:.1f} ms")
    print(f"cfunc time: {cfunc_time*1000:.1f} ms")
    print(f"Slowdown: {slowdown:.1f}×")
    
    # We allow up to 100× slowdown (JIT compilation overhead is acceptable)
    performance_ok = slowdown < 100.0
    
    if performance_ok:
        print("✓ Performance acceptable")
    else:
        print("✗ Performance too slow (>100× slower than SciPy)")
    
    return performance_ok

def main():
    """Run all automatic validation tests"""
    print("AUTOMATIC VALIDATION FOR BISPLREP/BISPLEV CFUNC")
    print("This test suite MUST pass for the implementation to be considered working")
    print("=" * 80)
    
    exact_match_ok = test_exact_match_requirement()
    boundary_ok = test_boundary_conditions()
    performance_ok = test_performance_requirement()
    
    print("\n" + "=" * 80)
    print("AUTOMATIC VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"Exact Match Test:    {'✓ PASS' if exact_match_ok else '✗ FAIL'}")
    print(f"Boundary Test:       {'✓ PASS' if boundary_ok else '✗ FAIL'}")
    print(f"Performance Test:    {'✓ PASS' if performance_ok else '✗ FAIL'}")
    
    all_passed = exact_match_ok and boundary_ok and performance_ok
    
    if all_passed:
        print("\n🎉 ALL AUTOMATIC VALIDATION TESTS PASSED!")
        print("The bisplrep/bisplev cfunc implementation is working correctly.")
        return 0
    else:
        print("\n❌ AUTOMATIC VALIDATION FAILED!")
        print("The implementation has critical issues that MUST be fixed.")
        print("This is a blocking failure - no further development should proceed")
        print("until these core requirements are met.")
        return 1

if __name__ == "__main__":
    sys.exit(main())