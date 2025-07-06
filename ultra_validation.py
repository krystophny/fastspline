#!/usr/bin/env python3
"""
Comprehensive validation of ultra-optimized DIERCKX cfunc implementations
Compare against DIERCKX f2py with floating point accuracy
"""

import numpy as np
import sys

# Import ultra-optimized cfunc implementations
from dierckx_numba_ultra import (
    fpback_ultra, fpgivs_ultra, fprota_ultra, fprati_ultra, fpbspl_ultra,
    warmup_ultra_functions
)

# Import DIERCKX f2py reference
import dierckx_f2py_fixed as dierckx_f2py

def colored_print(text, color="", style=""):
    """Print text (simplified without colorama)"""
    print(text)

def validate_fpback_ultra():
    """Comprehensive validation of ultra-optimized fpback"""
    colored_print("=" * 80)
    colored_print("                    1. FPBACK ULTRA - Backward Substitution")
    colored_print("=" * 80)
    
    max_error = 0.0
    test_cases = [
        (5, 3), (8, 2), (10, 4), (15, 5), (20, 6), (25, 3), (30, 7), (50, 8)
    ]
    
    for n, k in test_cases:
        nest = n + 10
        
        # Create well-conditioned test matrix
        a = np.zeros((nest, k), dtype=np.float64, order='F')
        for i in range(n):
            a[i, 0] = 2.0 + 0.1 * i  # Diagonal dominance
            for j in range(1, min(k, n-i)):
                a[i, j] = 0.1 / (j + 1)  # Upper triangular
        
        # Multiple test vectors
        test_vectors = [
            np.ones(n, dtype=np.float64),
            np.random.randn(n).astype(np.float64),
            np.linspace(1, n, n, dtype=np.float64),
            np.array([(-1)**i * (i+1) for i in range(n)], dtype=np.float64)
        ]
        
        for vec_idx, z in enumerate(test_vectors):
            # DIERCKX f2py reference
            a32 = a.astype(np.float32, order='F')
            z32 = z.astype(np.float32)
            c_ref = dierckx_f2py.fpback(a32, z32, n, k, nest)
            
            # Ultra-optimized cfunc implementation
            c_ultra = np.zeros(n, dtype=np.float64)
            fpback_ultra(a, z, n, k, c_ultra, nest)
            
            # Compare results
            error = np.max(np.abs(c_ultra.astype(np.float32) - c_ref))
            max_error = max(max_error, error)
            
            if error < 1e-6:
                colored_print(f"✓   n={n:2d}, k={k}, vec{vec_idx+1}: PASS")
            else:
                colored_print(f"✗   n={n:2d}, k={k}, vec{vec_idx+1}: FAIL (error={error:.2e})")
    
    colored_print(f"\n  Maximum error across all tests: {max_error:.2e}")
    return max_error

def validate_fpgivs_ultra():
    """Comprehensive validation of ultra-optimized fpgivs"""
    colored_print("\n" + "=" * 80)
    colored_print("                      2. FPGIVS ULTRA - Givens Rotations")
    colored_print("=" * 80)
    
    max_error = 0.0
    
    # Comprehensive test cases including edge cases
    test_cases = [
        # Standard cases
        (3.0, 4.0), (1.0, 0.0), (0.0, 1.0), (-2.0, 3.0),
        # Extreme values
        (1e-10, 1.0), (1.0, 1e-10), (100000.0, 1e-5), (-1e-8, 1e8),
        # Near-zero cases
        (1e-15, 1e-14), (0.0, 0.0), (1e-300, 1e-299),
        # Large values
        (1e10, 1e9), (-1e12, 1e11), (1e15, -1e14),
        # Equal magnitude cases
        (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0),
    ]
    
    # Add random cases
    for _ in range(10):
        test_cases.append((np.random.randn(), np.random.randn()))
    
    for piv, ww in test_cases:
        # DIERCKX f2py reference
        ref_result = dierckx_f2py.fpgivs(float(piv), float(ww))
        dd_ref, cos_ref, sin_ref = ref_result
        
        # Ultra-optimized cfunc implementation
        dd_ultra, cos_ultra, sin_ultra = fpgivs_ultra(float(piv), float(ww))
        
        # Compare results
        error_dd = abs(dd_ultra - dd_ref)
        error_cos = abs(cos_ultra - cos_ref)
        error_sin = abs(sin_ultra - sin_ref)
        error = max(error_dd, error_cos, error_sin)
        max_error = max(max_error, error)
        
        if error < 1e-6:
            colored_print(f"✓   piv={piv:g}, ww={ww:g}: PASS", Fore.GREEN)
        else:
            colored_print(f"✗   piv={piv:g}, ww={ww:g}: FAIL (error={error:.2e})", Fore.RED)
    
    colored_print(f"\n  Maximum error across all tests: {max_error:.2e}", Fore.WHITE)
    return max_error

def validate_fprota_ultra():
    """Comprehensive validation of ultra-optimized fprota"""
    colored_print("\n" + "=" * 80, Fore.BLUE, Style.BRIGHT)
    colored_print("                       3. FPROTA ULTRA - Apply Rotation", Fore.BLUE, Style.BRIGHT)
    colored_print("=" * 80, Fore.BLUE, Style.BRIGHT)
    
    max_error = 0.0
    
    test_cases = [
        # Identity rotations
        (1.0, 0.0, 3.0, 4.0), (0.0, 1.0, 2.0, -1.0),
        # Standard rotations
        (0.6, 0.8, 5.0, 3.0), (0.8, -0.6, 5.0, -3.0),
        (0.707, 0.707, 1.0, 1.0), (-0.707, 0.707, 2.0, -2.0),
        # Extreme cases
        (1.0, 0.0, 1e10, 1e-10), (0.0, 1.0, 1e-15, 1e15),
        # Zero cases
        (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
    ]
    
    # Add random normalized rotations
    for θ in np.linspace(0, 2*np.pi, 10):
        test_cases.append((np.cos(θ), np.sin(θ), np.random.randn(), np.random.randn()))
    
    for cos, sin, a, b in test_cases:
        # DIERCKX f2py reference
        ref_result = dierckx_f2py.fprota(cos, sin, a, b)
        a_ref, b_ref = ref_result
        
        # Ultra-optimized cfunc implementation
        a_ultra, b_ultra = fprota_ultra(cos, sin, a, b)
        
        # Compare results
        error_a = abs(a_ultra - a_ref)
        error_b = abs(b_ultra - b_ref)
        error = max(error_a, error_b)
        max_error = max(max_error, error)
        
        if error < 1e-6:
            colored_print(f"✓   rotation: PASS", Fore.GREEN)
        else:
            colored_print(f"✗   rotation: FAIL (error={error:.2e})", Fore.RED)
    
    colored_print(f"\n  Maximum error across all tests: {max_error:.2e}", Fore.WHITE)
    return max_error

def validate_fprati_ultra():
    """Comprehensive validation of ultra-optimized fprati"""
    colored_print("\n" + "=" * 80, Fore.BLUE, Style.BRIGHT)
    colored_print("                     4. FPRATI ULTRA - Rational Interpolation", Fore.BLUE, Style.BRIGHT)
    colored_print("=" * 80, Fore.BLUE, Style.BRIGHT)
    
    max_error = 0.0
    
    test_cases = [
        # Standard cases
        (1.0, 2.0, 2.0, 1.0, 3.0, -1.0),
        (0.1, 1.1, 0.5, 0.5, 0.9, -0.1),
        (10.0, 100.0, 5.0, 25.0, 1.0, 1.0),
        (-1.0, 2.0, 0.0, 0.0, 1.0, -2.0),
        # Infinity case (p3 <= 0)
        (1.0, 2.0, 2.0, 1.0, 0.0, -1.0),
        (0.5, 1.5, 1.5, 0.5, -1.0, 0.5),
        # Edge cases
        (0.0, 0.0, 1.0, 1.0, 2.0, 2.0),
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        # Large values
        (1e6, 1e7, 5e5, 2.5e6, 1e3, 1e4),
        # Small values
        (1e-6, 2e-6, 2e-6, 1e-6, 3e-6, -1e-6),
    ]
    
    # Add random cases
    for _ in range(5):
        test_cases.append((np.random.rand()*10-5, np.random.rand()*10-5, 
                          np.random.rand()*10-5, np.random.rand()*10-5,
                          np.random.rand()*10-5, np.random.rand()*10-5))
    
    for p1, f1, p2, f2, p3, f3 in test_cases:
        # DIERCKX f2py reference
        ref_result = dierckx_f2py.fprati(p1, f1, p2, f2, p3, f3)
        p_ref, p1_ref, f1_ref, p2_ref, f2_ref = ref_result
        
        # Ultra-optimized cfunc implementation
        p_ultra, p1_ultra, f1_ultra, p2_ultra, f2_ultra = fprati_ultra(p1, f1, p2, f2, p3, f3)
        
        # Compare results
        error_p = abs(p_ultra - p_ref)
        error_p1 = abs(p1_ultra - p1_ref)
        error_f1 = abs(f1_ultra - f1_ref)
        error_p2 = abs(p2_ultra - p2_ref)
        error_f2 = abs(f2_ultra - f2_ref)
        error = max(error_p, error_p1, error_f1, error_p2, error_f2)
        max_error = max(max_error, error)
        
        if error < 1e-6:
            colored_print(f"✓   case: PASS", Fore.GREEN)
        else:
            colored_print(f"✗   case: FAIL (error={error:.2e})", Fore.RED)
    
    colored_print(f"\n  Maximum error across all tests: {max_error:.2e}", Fore.WHITE)
    return max_error

def validate_fpbspl_ultra():
    """Comprehensive validation of ultra-optimized fpbspl"""
    colored_print("\n" + "=" * 80, Fore.BLUE, Style.BRIGHT)
    colored_print("                      5. FPBSPL ULTRA - B-spline Evaluation", Fore.BLUE, Style.BRIGHT)
    colored_print("=" * 80, Fore.BLUE, Style.BRIGHT)
    
    max_error = 0.0
    
    # Test all degrees from 1 to 5
    for k in range(1, 6):
        colored_print(f"\n  Testing degree k={k}:", Fore.CYAN)
        
        # Create comprehensive knot vectors
        knot_configs = [
            # Uniform knots
            np.concatenate([np.zeros(k+1), np.linspace(0, 1, 8), np.ones(k+1)]),
            # Non-uniform knots
            np.concatenate([np.zeros(k+1), [0.1, 0.3, 0.7, 0.9], np.ones(k+1)]),
            # Multiple knots
            np.concatenate([np.zeros(k+1), [0.25, 0.25, 0.5, 0.5, 0.75, 0.75], np.ones(k+1)])
        ]
        
        for config_idx, t in enumerate(knot_configs):
            t = t.astype(np.float64)
            n = len(t)
            
            # Test multiple evaluation points
            x_values = [0.1, 0.25, 0.5, 0.75, 0.9]
            
            for x in x_values:
                # Find appropriate l value (interval index)
                l = k + 1
                while l < n - k and x >= t[l]:
                    l += 1
                
                # DIERCKX f2py reference
                t32 = t.astype(np.float32)
                ref_result = dierckx_f2py.fpbspl(t32, k, x, l)
                h_ref = np.array(ref_result, dtype=np.float64)
                
                # Ultra-optimized cfunc implementation
                h_ultra = fpbspl_ultra(t, n, k, x, l)
                
                # Compare results (only first k+1 elements are meaningful)
                h_ref_trimmed = h_ref[:k+1] if len(h_ref) > k+1 else h_ref
                h_ultra_trimmed = h_ultra[:k+1]
                
                # Pad shorter array if needed
                if len(h_ref_trimmed) < len(h_ultra_trimmed):
                    h_ref_padded = np.zeros(len(h_ultra_trimmed))
                    h_ref_padded[:len(h_ref_trimmed)] = h_ref_trimmed
                    h_ref_trimmed = h_ref_padded
                elif len(h_ultra_trimmed) < len(h_ref_trimmed):
                    h_ultra_padded = np.zeros(len(h_ref_trimmed))
                    h_ultra_padded[:len(h_ultra_trimmed)] = h_ultra_trimmed
                    h_ultra_trimmed = h_ultra_padded
                
                error = np.max(np.abs(h_ultra_trimmed - h_ref_trimmed))
                max_error = max(max_error, error)
                
                if error < 1e-6:
                    colored_print(f"✓     x={x}, l={l}: PASS", Fore.GREEN)
                else:
                    colored_print(f"✗     x={x}, l={l}: FAIL (error={error:.2e})", Fore.RED)
    
    colored_print(f"\n  Maximum error across all tests: {max_error:.2e}", Fore.WHITE)
    return max_error

def main():
    """Run comprehensive ultra-optimization validation"""
    colored_print("COMPREHENSIVE ULTRA-OPTIMIZED DIERCKX CFUNC VALIDATION", Fore.WHITE, Style.BRIGHT)
    colored_print("MAXIMUM PERFORMANCE WITH FLOATING POINT ACCURACY", Fore.WHITE, Style.BRIGHT)
    
    # Warmup functions first
    warmup_ultra_functions()
    
    # Run all validation tests
    errors = []
    
    errors.append(validate_fpback_ultra())
    errors.append(validate_fpgivs_ultra())
    errors.append(validate_fprota_ultra())
    errors.append(validate_fprati_ultra())
    errors.append(validate_fpbspl_ultra())
    
    # Summary
    colored_print("\n" + "=" * 80, Fore.BLUE, Style.BRIGHT)
    colored_print("                             VALIDATION SUMMARY", Fore.BLUE, Style.BRIGHT)
    colored_print("=" * 80, Fore.BLUE, Style.BRIGHT)
    
    function_names = ["fpback", "fpgivs", "fprota", "fprati", "fpbspl"]
    all_passed = True
    
    for name, error in zip(function_names, errors):
        if error < 1e-6:
            colored_print(f"✓ {name:8s} : VALIDATED", Fore.GREEN)
        else:
            colored_print(f"✗ {name:8s} : FAILED (max error: {error:.2e})", Fore.RED)
            all_passed = False
    
    if all_passed:
        colored_print("\n🎉 ALL ULTRA-OPTIMIZED CFUNC ROUTINES VALIDATED SUCCESSFULLY!", Fore.GREEN, Style.BRIGHT)
        colored_print("Ultra-optimized cfunc implementations match DIERCKX reference.", Fore.GREEN)
        colored_print("Maximum performance achieved with floating point accuracy.", Fore.GREEN)
    else:
        colored_print("\n❌ SOME VALIDATIONS FAILED!", Fore.RED, Style.BRIGHT)
        colored_print("Ultra-optimization may have introduced errors.", Fore.RED)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())