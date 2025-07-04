#!/usr/bin/env python3
"""
Rigorous accuracy testing for ultra-optimized bisplev_cfunc_ultra vs SciPy.

This test ensures machine precision agreement before benchmarking performance.
"""

import numpy as np
import time
from scipy.interpolate import bisplrep, bisplev
from fastspline import bisplev_scalar

def test_accuracy_comprehensive():
    """Comprehensive accuracy test across multiple scenarios."""
    
    print("Ultra-Optimized bisplev_cfunc_ultra Accuracy Validation")
    print("=" * 60)
    
    test_cases = [
        ("Smooth function", lambda x, y: np.exp(-(x**2 + y**2)) * np.cos(np.pi * x) * np.sin(np.pi * y)),
        ("Polynomial", lambda x, y: x**2 + x*y + y**2 + 0.1*x**3 + 0.05*y**3),
        ("Oscillatory", lambda x, y: np.sin(5*x) * np.cos(5*y) + 0.1*np.sin(20*x*y)),
        ("Sharp gradient", lambda x, y: np.tanh(5*(x**2 + y**2 - 0.5))),
        ("Constant", lambda x, y: np.ones_like(x) * 3.14159)
    ]
    
    data_sizes = [100, 200, 500, 1000]
    
    all_passed = True
    
    for func_name, func in test_cases:
        print(f"\nTesting: {func_name}")
        print("-" * 40)
        
        for n_points in data_sizes:
            # Generate test data
            np.random.seed(42)  # Reproducible
            x_train = np.random.uniform(-1, 1, n_points)
            y_train = np.random.uniform(-1, 1, n_points)
            z_train = func(x_train, y_train)
            
            try:
                # Create SciPy spline
                tck = bisplrep(x_train, y_train, z_train, kx=3, ky=3, s=0)
                
                # Generate evaluation points
                n_eval = 200
                x_eval = np.random.uniform(-0.8, 0.8, n_eval)
                y_eval = np.random.uniform(-0.8, 0.8, n_eval)
                
                # Evaluate with all three methods
                scipy_results = []
                fast_results = []
                ultra_results = []
                
                for i in range(n_eval):
                    scipy_val = bisplev(x_eval[i], y_eval[i], tck)
                    fast_val = bisplev_cfunc(x_eval[i], y_eval[i], tck[0], tck[1], tck[2], 3, 3, len(tck[0]), len(tck[1]))
                    ultra_val = bisplev_cfunc_ultra(x_eval[i], y_eval[i], tck[0], tck[1], tck[2], 3, 3, len(tck[0]), len(tck[1]))
                    
                    scipy_results.append(scipy_val)
                    fast_results.append(fast_val)
                    ultra_results.append(ultra_val)
                
                # Calculate differences
                scipy_arr = np.array(scipy_results)
                fast_arr = np.array(fast_results)
                ultra_arr = np.array(ultra_results)
                
                diff_fast = np.abs(scipy_arr - fast_arr)
                diff_ultra = np.abs(scipy_arr - ultra_arr)
                
                max_diff_fast = np.max(diff_fast)
                max_diff_ultra = np.max(diff_ultra)
                rms_diff_fast = np.sqrt(np.mean(diff_fast**2))
                rms_diff_ultra = np.sqrt(np.mean(diff_ultra**2))
                
                # Check accuracy (realistic tolerance for numerical precision)
                tolerance = 1e-10
                fast_accurate = max_diff_fast < tolerance
                ultra_accurate = max_diff_ultra < tolerance
                
                status_fast = "✓ PASS" if fast_accurate else "✗ FAIL"
                status_ultra = "✓ PASS" if ultra_accurate else "✗ FAIL"
                
                print(f"  {n_points:4d} points: Fast {status_fast} (max={max_diff_fast:.2e}) | Ultra {status_ultra} (max={max_diff_ultra:.2e})")
                
                if not fast_accurate or not ultra_accurate:
                    all_passed = False
                    print(f"    WARNING: Accuracy below tolerance ({tolerance:.0e})")
                    
            except Exception as e:
                print(f"  {n_points:4d} points: ERROR - {e}")
                all_passed = False
    
    print(f"\nOverall Result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    return all_passed

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    
    print(f"\nEdge Case Testing")
    print("-" * 30)
    
    # Small coefficient arrays
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 0.5, 1.0]) 
    z = np.array([1.0, 2.0, 1.0, 3.0, 4.0, 3.0, 1.0, 2.0, 1.0])
    
    try:
        tck = bisplrep(x, y, z, kx=1, ky=1, s=0)  # Linear splines
        
        # Test evaluation points
        test_points = [(0.25, 0.25), (0.75, 0.75), (0.0, 0.0), (1.0, 1.0)]
        
        for x_eval, y_eval in test_points:
            scipy_val = bisplev(x_eval, y_eval, tck)
            ultra_val = bisplev_cfunc_ultra(x_eval, y_eval, tck[0], tck[1], tck[2], 1, 1, len(tck[0]), len(tck[1]))
            
            diff = abs(scipy_val - ultra_val)
            print(f"  Point ({x_eval}, {y_eval}): diff = {diff:.2e}")
            
    except Exception as e:
        print(f"  Edge case test failed: {e}")

def benchmark_performance():
    """Quick performance comparison."""
    
    print(f"\nPerformance Comparison")
    print("-" * 30)
    
    # Generate test data
    np.random.seed(42)
    n_points = 1000
    x_train = np.random.uniform(-1, 1, n_points)
    y_train = np.random.uniform(-1, 1, n_points) 
    z_train = x_train**2 + y_train**2
    
    try:
        tck = bisplrep(x_train, y_train, z_train, kx=3, ky=3, s=0)
        
        # Evaluation points
        n_eval = 5000
        x_eval = np.random.uniform(-0.8, 0.8, n_eval)
        y_eval = np.random.uniform(-0.8, 0.8, n_eval)
        
        # SciPy timing
        start = time.perf_counter()
        for i in range(n_eval):
            scipy_val = bisplev(x_eval[i], y_eval[i], tck)
        time_scipy = (time.perf_counter() - start) * 1000
        
        # FastSpline original timing
        start = time.perf_counter()
        for i in range(n_eval):
            fast_val = bisplev_cfunc(x_eval[i], y_eval[i], tck[0], tck[1], tck[2], 3, 3, len(tck[0]), len(tck[1]))
        time_fast = (time.perf_counter() - start) * 1000
        
        # FastSpline ultra timing
        start = time.perf_counter()
        for i in range(n_eval):
            ultra_val = bisplev_cfunc_ultra(x_eval[i], y_eval[i], tck[0], tck[1], tck[2], 3, 3, len(tck[0]), len(tck[1]))
        time_ultra = (time.perf_counter() - start) * 1000
        
        speedup_fast = time_scipy / time_fast
        speedup_ultra = time_scipy / time_ultra
        improvement = time_fast / time_ultra
        
        print(f"  SciPy:           {time_scipy:6.1f}ms")
        print(f"  FastSpline:      {time_fast:6.1f}ms  (speedup: {speedup_fast:.2f}x)")
        print(f"  FastSpline Ultra: {time_ultra:6.1f}ms  (speedup: {speedup_ultra:.2f}x)")
        print(f"  Ultra improvement: {improvement:.2f}x faster than original")
        
        if speedup_ultra > 1.0:
            print(f"  🚀 SUCCESS: Ultra version is {speedup_ultra:.2f}x faster than SciPy!")
        else:
            print(f"  ⚠️  Still {1/speedup_ultra:.2f}x slower than SciPy")
            
    except Exception as e:
        print(f"  Performance test failed: {e}")

if __name__ == "__main__":
    # Run comprehensive accuracy tests first
    accuracy_passed = test_accuracy_comprehensive()
    
    # Test edge cases
    test_edge_cases()
    
    # Only benchmark if accuracy tests pass
    if accuracy_passed:
        benchmark_performance()
    else:
        print("\n❌ Accuracy tests failed - skipping performance benchmark")
        print("Fix accuracy issues before performance testing!")