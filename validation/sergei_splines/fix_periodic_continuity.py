#!/usr/bin/env python3
"""
Fix periodic boundary continuity issues
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def analyze_periodic_issue():
    """Analyze the periodic boundary issue"""
    print("Analyzing Periodic Boundary Issue")
    print("=" * 50)
    
    # The issue is in the period calculation
    # For periodic splines with n points, the period should be:
    # - Regular splines: domain is [x_min, x_max] with n points, spacing h = (x_max - x_min)/(n-1)
    # - Periodic splines: domain is [x_min, x_max) with n points, spacing h = (x_max - x_min)/n
    
    print("\nRegular vs Periodic Grid:")
    print("-" * 30)
    
    n = 8
    x_min, x_max = 0.0, 1.0
    
    # Regular grid
    h_regular = (x_max - x_min) / (n - 1)
    x_regular = np.linspace(x_min, x_max, n)
    print(f"Regular (n={n}):")
    print(f"  h = {h_regular:.6f}")
    print(f"  x = {x_regular}")
    print(f"  Last point: x[{n-1}] = {x_regular[-1]:.6f}")
    
    # Periodic grid
    h_periodic = (x_max - x_min) / n
    x_periodic = np.linspace(x_min, x_max, n, endpoint=False)
    print(f"\nPeriodic (n={n}):")
    print(f"  h = {h_periodic:.6f}")
    print(f"  x = {x_periodic}")
    print(f"  Last point: x[{n-1}] = {x_periodic[-1]:.6f}")
    print(f"  Next would be: x[{n}] = {x_max:.6f} (wraps to x[0])")
    
    print("\nPeriod Calculation:")
    print("-" * 30)
    
    # Current (incorrect) period calculation
    period_wrong = h_regular * (n - 1)
    print(f"Current (wrong): period = h * (n-1) = {h_regular:.6f} * {n-1} = {period_wrong:.6f}")
    
    # Correct period calculation for periodic
    period_correct = x_max - x_min  # Always the domain size!
    print(f"Correct: period = x_max - x_min = {period_correct:.6f}")
    
    print("\nWrap-around Examples:")
    print("-" * 30)
    
    test_points = [0.95, 1.0, 1.05, 1.1, -0.05, -0.1]
    
    for x in test_points:
        # Current (wrong) wrapping
        xj_wrong = x - x_min
        if xj_wrong < 0:
            n_periods = int((-xj_wrong / period_wrong) + 1)
            xj_wrong = xj_wrong + period_wrong * n_periods
        elif xj_wrong >= period_wrong:
            n_periods = int(xj_wrong / period_wrong)
            xj_wrong = xj_wrong - period_wrong * n_periods
        
        # Correct wrapping
        xj_correct = (x - x_min) % period_correct
        
        print(f"  x = {x:6.2f}: wrong wrap = {xj_wrong:.6f}, correct wrap = {xj_correct:.6f}")

def test_fix():
    """Test the proposed fix"""
    print("\n\nTesting Proposed Fix")
    print("=" * 50)
    
    # The fix is to change the period calculation in evaluate_splines_1d_cfunc:
    # FROM: period = h_step * (num_points - 1)
    # TO:   period = h_step * num_points  (for periodic)
    
    print("Changes needed in sergei_splines.py:")
    print("-" * 40)
    print("1. In evaluate_splines_1d_cfunc (line ~527):")
    print("   Change: period = h_step * (num_points - 1)")
    print("   To:     period = h_step * num_points")
    print()
    print("2. In evaluate_splines_1d_der_cfunc (line ~762):")
    print("   Same change")
    print()
    print("3. In evaluate_splines_1d_der2_cfunc (line ~824):")
    print("   Same change")
    print()
    print("4. Also need to fix the construction step spacing:")
    print("   For periodic splines, h = (x_max - x_min) / n")
    print("   Not h = (x_max - x_min) / (n - 1)")

if __name__ == "__main__":
    analyze_periodic_issue()
    test_fix()
    
    print("\n" + "="*50)
    print("SOLUTION:")
    print("The period for periodic splines should be the full domain size.")
    print("Currently using (n-1)*h which is incorrect for periodic grids.")