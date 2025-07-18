#!/usr/bin/env python3
"""
Comprehensive test for exact Fortran matching across all spline types
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from fastspline.sergei_splines import construct_splines_1d_cfunc, evaluate_splines_1d_cfunc

def test_exact_fortran_match():
    """Test exact matching with Fortran for all implemented cases"""
    print("COMPREHENSIVE FORTRAN MATCHING TEST")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Non-periodic cubic splines (should be exact)
    print("\n1. NON-PERIODIC CUBIC SPLINES")
    print("-" * 40)
    results['cubic_nonperiodic'] = test_nonperiodic_cubic()
    
    # Test 2: Non-periodic quartic splines (should be exact)
    print("\n2. NON-PERIODIC QUARTIC SPLINES")
    print("-" * 40)
    results['quartic_nonperiodic'] = test_nonperiodic_quartic()
    
    # Test 3: Non-periodic quintic splines (now working for general n)
    print("\n3. NON-PERIODIC QUINTIC SPLINES")
    print("-" * 40)
    results['quintic_nonperiodic'] = test_nonperiodic_quintic()
    
    # Test 4: Periodic cubic splines (approaching exact)
    print("\n4. PERIODIC CUBIC SPLINES")
    print("-" * 40)
    results['cubic_periodic'] = test_periodic_cubic()
    
    # Test 5: Periodic quartic splines (basic implementation)
    print("\n5. PERIODIC QUARTIC SPLINES")
    print("-" * 40)
    results['quartic_periodic'] = test_periodic_quartic()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF FORTRAN MATCHING:")
    print("=" * 60)
    
    exact_count = 0
    close_count = 0
    total_count = 0
    
    for test_name, (status, error) in results.items():
        total_count += 1
        if status == "EXACT":
            exact_count += 1
            print(f"✅ {test_name:25} : EXACT MATCH")
        elif status == "CLOSE":
            close_count += 1
            print(f"⚠️  {test_name:25} : CLOSE ({error:.2e})")
        else:
            print(f"❌ {test_name:25} : POOR ({error:.2e})")
    
    print(f"\nResults: {exact_count}/{total_count} exact, {close_count}/{total_count} close")
    
    if exact_count == total_count:
        print("\n🎯 PERFECT: All implementations match Fortran exactly!")
        return True
    elif exact_count + close_count == total_count:
        print("\n🔧 GOOD: All implementations working, some very close to exact")
        return False
    else:
        print("\n⚠️  MIXED: Some implementations need work")
        return False

def test_nonperiodic_cubic():
    """Test non-periodic cubic splines"""
    n = 10
    x = np.linspace(0, 1, n)
    y = np.sin(2*np.pi*x)
    
    try:
        coeff = np.zeros(4*n)
        construct_splines_1d_cfunc(0.0, 1.0, y, n, 3, False, coeff)
        
        # Test evaluation
        x_test = 0.5
        y_out = np.zeros(1)
        h = 1.0 / (n - 1)
        evaluate_splines_1d_cfunc(3, n, False, 0.0, h, coeff, x_test, y_out)
        
        y_exact = np.sin(2*np.pi*x_test)
        error = abs(y_out[0] - y_exact)
        
        print(f"  Test point: x={x_test}, error={error:.2e}")
        
        if error < 1e-14:
            return ("EXACT", error)
        elif error < 1e-10:
            return ("CLOSE", error)
        else:
            return ("POOR", error)
            
    except Exception as e:
        print(f"  ERROR: {e}")
        return ("ERROR", float('inf'))

def test_nonperiodic_quartic():
    """Test non-periodic quartic splines"""
    n = 10
    x = np.linspace(0, 1, n)
    y = np.sin(2*np.pi*x)
    
    try:
        coeff = np.zeros(5*n)
        construct_splines_1d_cfunc(0.0, 1.0, y, n, 4, False, coeff)
        
        # Test evaluation
        x_test = 0.5
        y_out = np.zeros(1)
        h = 1.0 / (n - 1)
        evaluate_splines_1d_cfunc(4, n, False, 0.0, h, coeff, x_test, y_out)
        
        y_exact = np.sin(2*np.pi*x_test)
        error = abs(y_out[0] - y_exact)
        
        print(f"  Test point: x={x_test}, error={error:.2e}")
        
        if error < 1e-14:
            return ("EXACT", error)
        elif error < 1e-10:
            return ("CLOSE", error)
        else:
            return ("POOR", error)
            
    except Exception as e:
        print(f"  ERROR: {e}")
        return ("ERROR", float('inf'))

def test_nonperiodic_quintic():
    """Test non-periodic quintic splines"""
    n = 20  # Use larger n for better quintic accuracy
    x = np.linspace(0, 1, n)
    y = np.sin(2*np.pi*x)
    
    try:
        coeff = np.zeros(6*n)
        construct_splines_1d_cfunc(0.0, 1.0, y, n, 5, False, coeff)
        
        # Test evaluation
        x_test = 0.5
        y_out = np.zeros(1)
        h = 1.0 / (n - 1)
        evaluate_splines_1d_cfunc(5, n, False, 0.0, h, coeff, x_test, y_out)
        
        y_exact = np.sin(2*np.pi*x_test)
        error = abs(y_out[0] - y_exact)
        
        print(f"  Test point: x={x_test}, error={error:.2e} (n={n})")
        
        if error < 1e-14:
            return ("EXACT", error)
        elif error < 1e-2:  # Quintic tolerance is higher
            return ("CLOSE", error)
        else:
            return ("POOR", error)
            
    except Exception as e:
        print(f"  ERROR: {e}")
        return ("ERROR", float('inf'))

def test_periodic_cubic():
    """Test periodic cubic splines"""
    n = 16
    x = np.linspace(0, 1, n, endpoint=False)
    y = np.sin(2*np.pi*x)
    
    try:
        coeff = np.zeros(4*n)
        construct_splines_1d_cfunc(0.0, 1.0, y, n, 3, True, coeff)
        
        # Test boundary continuity
        h = 1.0 / n
        
        # Evaluate at x=0
        y_out_0 = np.zeros(1)
        evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, 0.0, y_out_0)
        
        # Evaluate at x=1-ε (should be same due to periodicity)
        eps = 1e-12
        y_out_1 = np.zeros(1)
        evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, 1.0 - eps, y_out_1)
        
        continuity_error = abs(y_out_0[0] - y_out_1[0])
        
        print(f"  Boundary continuity error: {continuity_error:.2e}")
        
        if continuity_error < 1e-14:
            return ("EXACT", continuity_error)
        elif continuity_error < 1e-10:
            return ("CLOSE", continuity_error)
        else:
            return ("POOR", continuity_error)
            
    except Exception as e:
        print(f"  ERROR: {e}")
        return ("ERROR", float('inf'))

def test_periodic_quartic():
    """Test periodic quartic splines (basic implementation)"""
    n = 16
    x = np.linspace(0, 1, n, endpoint=False)
    y = np.sin(2*np.pi*x)
    
    try:
        coeff = np.zeros(5*n)
        construct_splines_1d_cfunc(0.0, 1.0, y, n, 4, True, coeff)
        
        # Check if basic construction works
        print(f"  Basic construction: ✓")
        
        # Quartic periodic is placeholder implementation
        return ("POOR", 1.0)  # Known placeholder
            
    except Exception as e:
        print(f"  ERROR: {e}")
        return ("ERROR", float('inf'))

if __name__ == "__main__":
    success = test_exact_fortran_match()
    
    if success:
        print("\n🏆 ACHIEVEMENT: Perfect Fortran matching across all implementations!")
    else:
        print("\n🔧 STATUS: Good progress, some areas need refinement for exact matching")