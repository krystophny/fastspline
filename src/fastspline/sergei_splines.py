#!/usr/bin/env python3
"""
Sergei's splines - Pure Numba cfunc implementation
Complete port from interpolate.f90 with both 1D and 2D support
Final version with proper pointer arithmetic
"""

import numpy as np
from numba import cfunc, types
import ctypes

# Constants for quintic spline
RHOP = 23.246950765959598  # 13.0 + sqrt(105.0)
RHOM = 2.753049234040402   # 13.0 - sqrt(105.0)

# ==== 1D SPLINE CONSTRUCTION WITH INLINED ALGORITHMS ====
@cfunc(types.void(
    types.float64,                      # x_min
    types.float64,                      # x_max
    types.CPointer(types.float64),      # y values
    types.int32,                        # num_points
    types.int32,                        # order
    types.int32,                        # periodic (0 or 1)
    types.CPointer(types.float64),      # output coeff array ((order+1) * num_points)
), nopython=True, nogil=True, cache=True, fastmath=True)
def construct_splines_1d_cfunc(x_min, x_max, y, num_points, order, periodic, coeff):
    """Construct 1D spline - complete implementation with inlined algorithms"""
    
    if periodic:
        h_step = (x_max - x_min) / num_points
    else:
        h_step = (x_max - x_min) / (num_points - 1)
    n = num_points
    
    # Copy y values to first row of coefficient array
    for i in range(n):
        coeff[i] = y[i]
    
    if order == 3:
        # CUBIC SPLINE IMPLEMENTATION
        if periodic == 0:
            # REGULAR CUBIC SPLINE (natural boundary conditions)
            # Working arrays (dynamically allocated)
            al_array = np.zeros(num_points, dtype=np.float64)
            bt_array = np.zeros(num_points, dtype=np.float64)
            
            # Initialize
            ak1 = 0.0
            ak2 = 0.0
            am1 = 0.0
            am2 = 0.0
            k = n - 1
            al_array[0] = ak1
            bt_array[0] = am1
            n2 = n - 2
            c = -4.0 * h_step
            
            # Forward elimination
            for i in range(n2):
                e = -3.0 * ((coeff[i+2] - coeff[i+1]) - (coeff[i+1] - coeff[i])) / h_step
                c1 = c - al_array[i] * h_step
                if abs(c1) > 1e-15:
                    al_array[i+1] = h_step / c1
                    bt_array[i+1] = (h_step * bt_array[i] + e) / c1
                else:
                    al_array[i+1] = 0.0
                    bt_array[i+1] = 0.0
            
            # Back substitution for c coefficients
            # FORTRAN: ci(n) = (am2+ak2*bt(k))/(1.d0-al(k)*ak2)
            denom = 1.0 - al_array[k-1] * ak2
            if abs(denom) > 1e-15:
                coeff[2*n + n-1] = (am2 + ak2 * bt_array[k-1]) / denom
            else:
                coeff[2*n + n-1] = 0.0
                
            # FORTRAN: DO i = 1,k: i5 = n-i; ci(i5) = al(i5)*ci(i5+1)+bt(i5)
            for i in range(1, k+1):
                i5 = n - i
                coeff[2*n + i5-1] = al_array[i5-1] * coeff[2*n + i5] + bt_array[i5-1]
            
            # Calculate b and d coefficients
            n2 = n - 1
            for i in range(n2):
                coeff[n + i] = (coeff[i+1] - coeff[i]) / h_step - h_step * (coeff[2*n + i+1] + 2.0 * coeff[2*n + i]) / 3.0
                coeff[3*n + i] = (coeff[2*n + i+1] - coeff[2*n + i]) / h_step / 3.0
            
            coeff[n + n-1] = 0.0
            coeff[3*n + n-1] = 0.0
            
        else:
            # Periodic cubic spline - EXACT 1:1 port from Fortran splper subroutine
            # From spl_three_to_five.f90, lines 473-544
            
            # Allocate working arrays
            bmx = np.zeros(n, dtype=np.float64)
            yl = np.zeros(n, dtype=np.float64) 
            amx1 = np.zeros(n, dtype=np.float64)
            amx2 = np.zeros(n, dtype=np.float64)
            amx3 = np.zeros(n, dtype=np.float64)
            
            # FORTRAN: bmx(1) = 1.d30
            bmx[0] = 1e30
            
            # FORTRAN variables
            nmx = n - 1
            n1 = nmx - 1
            n2 = nmx - 2
            psi = 3.0 / (h_step * h_step)
            
            # CALL spfper(n,amx1,amx2,amx3) - inline implementation
            # From lines 547-584 in Fortran
            nn = nmx  # n in spfper corresponds to nmx here
            nn1 = nn - 1
            
            amx1[0] = 2.0
            amx2[0] = 0.5
            amx3[0] = 0.5
            amx1[1] = np.sqrt(15.0) / 2.0
            amx2[1] = 1.0 / amx1[1]
            amx3[1] = -0.25 / amx1[1]
            beta = 3.75
            
            for i in range(2, nn1):
                i1 = i - 1
                beta = 4.0 - 1.0 / beta
                amx1[i] = np.sqrt(beta)
                amx2[i] = 1.0 / amx1[i]
                amx3[i] = -amx3[i1] / amx1[i] / amx1[i1]
            
            amx3[nn1-1] = amx3[nn1-1] + 1.0 / amx1[nn1-1]
            amx2[nn1-1] = amx3[nn1-1]
            ss = 0.0
            for i in range(nn1):
                ss = ss + amx3[i] * amx3[i]
            amx1[nn-1] = np.sqrt(4.0 - ss)
            
            # FORTRAN: bmx(nmx) = (y(nmx+1)-2.d0*y(nmx)+y(nmx-1))*psi
            # y(nmx+1) is periodic so it's y(1), which is coeff[0] in Python
            bmx[nmx-1] = (coeff[0] - 2.0*coeff[nmx-1] + coeff[nmx-2]) * psi
            
            # FORTRAN: bmx(1) = (y(2)-y(1)-y(nmx+1)+y(nmx))*psi
            bmx[0] = (coeff[1] - coeff[0] - coeff[0] + coeff[nmx-1]) * psi
            
            # FORTRAN: DO i = 3,nmx: bmx(i-1) = (y(i)-2.d0*y(i-1)+y(i-2))*psi
            for i in range(2, nmx):
                # Fortran i goes from 3 to nmx, so bmx(i-1) means bmx[i-2] in Python
                bmx[i-1] = (coeff[i] - 2.0*coeff[i-1] + coeff[i-2]) * psi
            
            # FORTRAN: yl(1) = bmx(1)/amx1(1)
            yl[0] = bmx[0] / amx1[0]
            
            # FORTRAN: DO i = 2,n1
            for i in range(1, n1):
                i1 = i - 1
                yl[i] = (bmx[i] - yl[i1]*amx2[i1]) / amx1[i]
            
            # FORTRAN: ss = 0.d0; DO i = 1,n1: ss = ss+yl(i)*amx3(i)
            ss = 0.0
            for i in range(n1):
                ss = ss + yl[i] * amx3[i]
            
            # FORTRAN: yl(nmx) = (bmx(nmx)-ss)/amx1(nmx)
            yl[nmx-1] = (bmx[nmx-1] - ss) / amx1[nmx-1]
            
            # FORTRAN: bmx(nmx) = yl(nmx)/amx1(nmx)
            bmx[nmx-1] = yl[nmx-1] / amx1[nmx-1]
            
            # FORTRAN: bmx(n1) = (yl(n1)-amx2(n1)*bmx(nmx))/amx1(n1)
            bmx[n1-1] = (yl[n1-1] - amx2[n1-1]*bmx[nmx-1]) / amx1[n1-1]
            
            # FORTRAN: DO i = n2,1,-1
            for i in range(n2-1, -1, -1):
                bmx[i] = (yl[i] - amx3[i]*bmx[nmx-1] - amx2[i]*bmx[i+1]) / amx1[i]
            
            # FORTRAN: DO i = 1,nmx: ci(i) = bmx(i)
            for i in range(nmx):
                coeff[2*n + i] = bmx[i]
            
            # FORTRAN: DO i = 1,n1
            for i in range(n1):
                # bi(i) = (y(i+1)-y(i))/h-h*(ci(i+1)+2.d0*ci(i))/3.d0
                coeff[n + i] = (coeff[i+1] - coeff[i]) / h_step - h_step * (coeff[2*n + i+1] + 2.0*coeff[2*n + i]) / 3.0
                # di(i) = (ci(i+1)-ci(i))/h/3.d0
                coeff[3*n + i] = (coeff[2*n + i+1] - coeff[2*n + i]) / h_step / 3.0
            
            # FORTRAN: bi(nmx) = (y(n)-y(n-1))/h-h*(ci(1)+2.d0*ci(nmx))/3.d0
            # y(n) wraps to y(1) for periodic, so it's coeff[0]
            coeff[n + nmx-1] = (coeff[0] - coeff[nmx-1]) / h_step - h_step * (coeff[2*n + 0] + 2.0*coeff[2*n + nmx-1]) / 3.0
            
            # FORTRAN: di(nmx) = (ci(1)-ci(nmx))/h/3.d0
            coeff[3*n + nmx-1] = (coeff[2*n + 0] - coeff[2*n + nmx-1]) / h_step / 3.0
            
            # FORTRAN: Fix of problems at upper periodicity boundary
            # bi(n) = bi(1), ci(n) = ci(1), di(n) = di(1)
            coeff[n + n-1] = coeff[n + 0]       # bi(n) = bi(1)
            coeff[2*n + n-1] = coeff[2*n + 0]   # ci(n) = ci(1)
            coeff[3*n + n-1] = coeff[3*n + 0]   # di(n) = di(1)
                
    elif order == 4:
        # Quartic spline - EXACT 1:1 port from Fortran spl_four_reg/spl_four_per
        if periodic == 0:
            # Regular quartic spline - EXACT 1:1 Fortran port from spl_four_reg
            # From spl_three_to_five.f90 lines 263-330
            
            # Working arrays
            alp = np.zeros(n, dtype=np.float64)
            bet = np.zeros(n, dtype=np.float64)
            gam = np.zeros(n, dtype=np.float64)
            
            # Arrays for coefficients (using 1-based indexing style)
            b = np.zeros(n, dtype=np.float64)
            c = np.zeros(n, dtype=np.float64)  
            d = np.zeros(n, dtype=np.float64)
            e = np.zeros(n, dtype=np.float64)
            
            # FORTRAN: Initial d and e calculation (lines 275-282)
            fpl31 = 0.5*(coeff[1] + coeff[3]) - coeff[2]  # a(2)+a(4)-a(3)
            fpl40 = 0.5*(coeff[0] + coeff[4]) - coeff[2]  # a(1)+a(5)-a(3)
            fmn31 = 0.5*(coeff[3] - coeff[1])  # a(4)-a(2)
            fmn40 = 0.5*(coeff[4] - coeff[0])  # a(5)-a(1)
            d[2] = (fmn40 - 2.0*fmn31) / 6.0  # d(3)
            e[2] = (fpl40 - 4.0*fpl31) / 12.0  # e(3)
            d[1] = d[2] - 4.0*e[2]  # d(2)
            d[0] = d[2] - 8.0*e[2]  # d(1)
            
            # FORTRAN: First elimination (lines 284-291)
            alp[0] = 0.0  # alp(1)
            bet[0] = d[0] + d[1]  # bet(1) = d(1) + d(2)
            
            # do i=1,n-3
            for i in range(1, n-2):
                ip1 = i + 1
                alp[ip1-1] = -1.0 / (10.0 + alp[i-1])  # alp(ip1) 
                bet[ip1-1] = alp[ip1-1] * (bet[i-1] - 4.0*(coeff[i+2] - 3.0*(coeff[i+1] - coeff[i]) - coeff[i-1]))
            
            # FORTRAN: End boundary calculation (lines 293-302)
            fpl31 = 0.5*(coeff[n-4] + coeff[n-2]) - coeff[n-3]  # a(n-3)+a(n-1)-a(n-2)
            fpl40 = 0.5*(coeff[n-5] + coeff[n-1]) - coeff[n-3]  # a(n-4)+a(n)-a(n-2)
            fmn31 = 0.5*(coeff[n-2] - coeff[n-4])  # a(n-1)-a(n-3)
            fmn40 = 0.5*(coeff[n-1] - coeff[n-5])  # a(n)-a(n-4)
            d[n-3] = (fmn40 - 2.0*fmn31) / 6.0  # d(n-2)
            e[n-3] = (fpl40 - 4.0*fpl31) / 12.0  # e(n-2)
            d[n-2] = d[n-3] + 4.0*e[n-3]  # d(n-1)
            d[n-1] = d[n-3] + 8.0*e[n-3]  # d(n)
            
            # FORTRAN: Back substitution (lines 302-310)
            gam[n-2] = d[n-1] + d[n-2]  # gam(n-1) = d(n) + d(n-1)
            
            # do i=n-2,1,-1
            for i in range(n-2, 0, -1):
                gam[i-1] = gam[i]*alp[i-1] + bet[i-1]  # gam(i)
                d[i-1] = gam[i-1] - d[i]  # d(i) = gam(i) - d(i+1)
                e[i-1] = (d[i] - d[i-1]) / 4.0  # e(i) = (d(i+1) - d(i))/4
                c[i-1] = 0.5*(coeff[i+1] + coeff[i-1]) - coeff[i] - 0.125*(d[i+1] + 12.0*d[i] + 11.0*d[i-1])
                b[i-1] = coeff[i] - coeff[i-1] - c[i-1] - (3.0*d[i-1] + d[i]) / 4.0
            
            # FORTRAN: Final boundary coefficients (lines 312-317)
            b[n-2] = b[n-3] + 2.0*c[n-3] + 3.0*d[n-3] + 4.0*e[n-3]  # b(n-1)
            c[n-2] = c[n-3] + 3.0*d[n-3] + 6.0*e[n-3]  # c(n-1)
            e[n-2] = coeff[n-1] - coeff[n-2] - b[n-2] - c[n-2] - d[n-2]  # e(n-1)
            b[n-1] = b[n-2] + 2.0*c[n-2] + 3.0*d[n-2] + 4.0*e[n-2]  # b(n)
            c[n-1] = c[n-2] + 3.0*d[n-2] + 6.0*e[n-2]  # c(n)
            e[n-1] = e[n-2]  # e(n)
            
            # FORTRAN: Scaling (lines 319-326)
            fac = 1.0 / h_step
            b = b * fac
            fac = fac / h_step
            c = c * fac
            fac = fac / h_step
            d = d * fac
            fac = fac / h_step
            e = e * fac
            
            # Copy to coefficient array
            for i in range(n):
                coeff[n + i] = b[i]
                coeff[2*n + i] = c[i]
                coeff[3*n + i] = d[i]
                coeff[4*n + i] = e[i]
                
        else:
            # Periodic quartic spline - EXACT 1:1 Fortran port from spl_four_per
            # From spl_three_to_five.f90 lines 333-414
            
            # Working arrays (using 1-based indexing mapping)
            alp = np.zeros(n+1, dtype=np.float64)  # alp(1:n)
            bet = np.zeros(n+1, dtype=np.float64)  # bet(1:n)
            gam = np.zeros(n+1, dtype=np.float64)  # gam(1:n)
            
            # Arrays for coefficients
            b = np.zeros(n, dtype=np.float64)
            c = np.zeros(n, dtype=np.float64)
            d = np.zeros(n, dtype=np.float64)
            e = np.zeros(n, dtype=np.float64)
            
            # FORTRAN: Base values for periodic correction (lines 345-346)
            base1 = -5.0 + 2.0*np.sqrt(6.0)
            base2 = -5.0 - 2.0*np.sqrt(6.0)
            
            # FORTRAN: First elimination (lines 348-359)
            alp[1] = 0.0  # alp(1)
            bet[1] = 0.0  # bet(1)
            
            # do i=1,n-3 (Fortran 1-based)
            for i in range(1, n-2):
                ip1 = i + 1
                alp[ip1] = -1.0 / (10.0 + alp[i])  # alp(ip1)
                expr = coeff[i+3-1] - 3.0*(coeff[i+2-1] - coeff[ip1-1]) - coeff[i-1]  # Convert to 0-based
                bet[ip1] = alp[ip1] * (bet[i] - 4.0*expr)
            
            # Special periodic boundary terms
            alp[n-1] = -1.0 / (10.0 + alp[n-2])  # alp(n-1)
            expr = coeff[2-1] - 3.0*(coeff[n-1] - coeff[n-1-1]) - coeff[n-2-1]  # Convert to 0-based
            bet[n-1] = alp[n-1] * (bet[n-2] - 4.0*expr)
            
            alp[n] = -1.0 / (10.0 + alp[n-1])  # alp(n)
            expr = coeff[3-1] - 3.0*(coeff[2-1] - coeff[n-1]) - coeff[n-1-1]  # Convert to 0-based
            bet[n] = alp[n] * (bet[n-1] - 4.0*expr)
            
            # FORTRAN: Back substitution (lines 361-365)
            gam[n] = bet[n]  # gam(n)
            
            for i in range(n-1, 0, -1):
                gam[i] = gam[i+1]*alp[i] + bet[i]  # gam(i)
            
            # FORTRAN: Periodic correction (lines 367-378)
            phi1 = (gam[n]*base2 + gam[2]) / (base2 - base1) / (1.0 - base1**(n-1))
            phi2 = (gam[n]*base1 + gam[2]) / (base2 - base1) / (1.0 - (1.0/base2)**(n-1))
            
            # Apply phi2 correction (backwards)
            phi2_temp = phi2
            for i in range(n, 0, -1):
                gam[i] = gam[i] + phi2_temp
                phi2_temp = phi2_temp / base2
            
            # Apply phi1 correction (forwards)  
            phi1_temp = phi1
            for i in range(1, n+1):
                gam[i] = gam[i] + phi1_temp
                phi1_temp = phi1_temp * base1
            
            # FORTRAN: d coefficients (lines 380-389)
            d[n-1] = 0.0  # d(n)
            for i in range(n-1, 0, -1):
                d[i-1] = gam[i] - d[i]  # d(i) = gam(i) - d(i+1)
            
            # Alternating correction
            phi = -0.5 * d[0]  # phi = -0.5*d(1)
            for i in range(n):
                d[i] = d[i] + phi
                phi = -phi
            
            # FORTRAN: e, c, b coefficients (lines 391-402)
            # e(n)=(d(2)-d(n))/4.d0
            e[n-1] = (d[1] - d[n-1]) / 4.0
            # c(n)=0.5d0*(a(3)+a(n))-a(2)-0.125d0*(d(3)+12.d0*d(2)+11.d0*d(n))
            c[n-1] = 0.5*(coeff[2] + coeff[n-1]) - coeff[1] - 0.125*(d[2] + 12.0*d[1] + 11.0*d[n-1])
            # b(n)=a(2)-a(n)-c(n)-(3.d0*d(n)+d(2))/4.d0
            b[n-1] = coeff[1] - coeff[n-1] - c[n-1] - (3.0*d[n-1] + d[1]) / 4.0
            
            # e(n-1)=(d(1)-d(n-1))/4.d0
            e[n-2] = (d[0] - d[n-2]) / 4.0
            # c(n-1)=0.5d0*(a(2)+a(n-1))-a(1)-0.125d0*(d(2)+12.d0*d(1)+11.d0*d(n-1))
            c[n-2] = 0.5*(coeff[1] + coeff[n-2]) - coeff[0] - 0.125*(d[1] + 12.0*d[0] + 11.0*d[n-2])
            # b(n-1)=a(1)-a(n-1)-c(n-1)-(3.d0*d(n-1)+d(1))/4.d0
            b[n-2] = coeff[0] - coeff[n-2] - c[n-2] - (3.0*d[n-2] + d[0]) / 4.0
            
            # do i=n-2,1,-1 (Fortran loop from n-2 down to 1, Python i=n-3 down to i=0)
            for i in range(n-3, -1, -1):
                # e(i)=(d(i+1)-d(i))/4.d0
                e[i] = (d[i+1] - d[i]) / 4.0
                
                if i == 0:
                    # Handle i=0 case (Fortran i=1) - EXACT FORTRAN MAPPING
                    # c(1)=0.5d0*(a(3)+a(1))-a(2)-0.125d0*(d(3)+12.d0*d(2)+11.d0*d(1))
                    c[i] = 0.5*(coeff[i+2] + coeff[i]) - coeff[i+1] - 0.125*(d[i+2] + 12.0*d[i+1] + 11.0*d[i])
                    # b(1)=a(2)-a(1)-c(1)-(3.d0*d(1)+d(2))/4.d0
                    b[i] = coeff[i+1] - coeff[i] - c[i] - (3.0*d[i] + d[i+1]) / 4.0
                else:
                    # Regular case for i >= 1
                    # c(i)=0.5d0*(a(i+2)+a(i))-a(i+1)-0.125d0*(d(i+2)+12.d0*d(i+1)+11.d0*d(i))
                    c[i] = 0.5*(coeff[i+2] + coeff[i]) - coeff[i+1] - 0.125*(d[i+2] + 12.0*d[i+1] + 11.0*d[i])
                    # b(i)=a(i+1)-a(i)-c(i)-(3.d0*d(i)+d(i+1))/4.d0
                    b[i] = coeff[i+1] - coeff[i] - c[i] - (3.0*d[i] + d[i+1]) / 4.0
            
            # FORTRAN: Scaling (lines 404-411)
            fac = 1.0 / h_step
            b = b * fac
            fac = fac / h_step
            c = c * fac
            fac = fac / h_step
            d = d * fac
            fac = fac / h_step
            e = e * fac
            
            # Copy to coefficient array
            for i in range(n):
                coeff[n + i] = b[i]
                coeff[2*n + i] = c[i]
                coeff[3*n + i] = d[i]
                coeff[4*n + i] = e[i]
            
    elif order == 5:
        # Quintic spline - EXACT 1:1 port from Fortran spl_five_reg/spl_five_per
        if periodic == 0:
            # Regular quintic spline - EXACT 1:1 Fortran port from spl_five_reg
            # From spl_three_to_five.f90 lines 7-144
            
            # Fortran constants
            rhop = 13.0 + np.sqrt(105.0)
            rhom = 13.0 - np.sqrt(105.0)
            
            # Working arrays
            alp = np.zeros(n, dtype=np.float64)
            bet = np.zeros(n, dtype=np.float64)
            gam = np.zeros(n, dtype=np.float64)
            
            # FORTRAN: First boundary system matrix (lines 23-32)
            a11 = 1.0
            a12 = 1.0/4.0
            a13 = 1.0/16.0
            a21 = 3.0
            a22 = 27.0/4.0
            a23 = 9.0*27.0/16.0
            a31 = 5.0
            a32 = 125.0/4.0
            a33 = 5.0**5/16.0
            det = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a12*a21*a33 - a13*a22*a31 - a11*a23*a32
            
            # FORTRAN: Beginning boundary (lines 33-41)
            b1 = coeff[3] - coeff[2]  # a(4)-a(3)
            b2 = coeff[4] - coeff[1]  # a(5)-a(2)
            b3 = coeff[5] - coeff[0]  # a(6)-a(1)
            bbeg = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32) / det
            dbeg = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3) / det
            fbeg = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32) / det
            
            # FORTRAN: End boundary (lines 42-50)
            b1 = coeff[n-3] - coeff[n-4]  # a(n-2)-a(n-3)
            b2 = coeff[n-2] - coeff[n-5]  # a(n-1)-a(n-4)
            b3 = coeff[n-1] - coeff[n-6]  # a(n)-a(n-5)
            bend = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32) / det
            dend = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3) / det
            fend = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32) / det
            
            # FORTRAN: Second boundary system matrix (lines 51-59)
            a11 = 2.0
            a12 = 1.0/2.0
            a13 = 1.0/8.0
            a21 = 2.0
            a22 = 9.0/2.0
            a23 = 81.0/8.0
            a31 = 2.0
            a32 = 25.0/2.0
            a33 = 625.0/8.0
            det = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a12*a21*a33 - a13*a22*a31 - a11*a23*a32
            
            # FORTRAN: Beginning boundary (lines 61-69)
            b1 = coeff[3] + coeff[2]  # a(4)+a(3)
            b2 = coeff[4] + coeff[1]  # a(5)+a(2)
            b3 = coeff[5] + coeff[0]  # a(6)+a(1)
            abeg = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32) / det
            cbeg = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3) / det
            ebeg = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32) / det
            
            # FORTRAN: End boundary (lines 70-78)
            b1 = coeff[n-3] + coeff[n-4]  # a(n-2)+a(n-3)
            b2 = coeff[n-2] + coeff[n-5]  # a(n-1)+a(n-4)
            b3 = coeff[n-1] + coeff[n-6]  # a(n)+a(n-5)
            aend = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32) / det
            cend = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3) / det
            eend = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32) / det
            
            # FORTRAN: First elimination (lines 82-90)
            # In Fortran: alp(1) = 0.0, bet(1) = ebeg*(2.0+rhom) - 5.0*fbeg*(3.0+1.5*rhom)
            # Direct mapping to Python: alp[1] = 0.0, bet[1] = ...
            alp[1] = 0.0
            bet[1] = ebeg*(2.0 + rhom) - 5.0*fbeg*(3.0 + 1.5*rhom)
            
            # do i=1,n-4
            for i in range(1, n-3):
                ip1 = i + 1
                # FORTRAN: alp(ip1) where ip1=i+1 
                # Direct mapping: when i=1, ip1=2, set alp[2]
                alp[ip1] = -1.0 / (rhop + alp[i])
                # FORTRAN: 5.d0*(a(i+4)-4.d0*a(i+3)+6.d0*a(i+2)-4.d0*a(ip1)+a(i))
                # Convert Fortran 1-based to Python 0-based: a(i) -> coeff[i-1]
                bet[ip1] = alp[ip1] * (bet[i] - 5.0*(coeff[i+3] - 4.0*coeff[i+2] + 6.0*coeff[i+1] - 4.0*coeff[i] + coeff[i-1]))
            
            # FORTRAN: Back substitution (lines 92-95)
            # gam(n-2)=eend*(2.d0+rhom)+5.d0*fend*(3.d0+1.5d0*rhom) !gamma
            gam[n-2] = eend*(2.0 + rhom) + 5.0*fend*(3.0 + 1.5*rhom)
            # do i=n-3,1,-1 (Fortran 1-based)
            # Fortran: gam(i) = gam(i+1)*alp(i) + bet(i)
            # Direct mapping: when i=3, calculate gam[3] using gam[4]
            for i in range(n-3, 0, -1):
                gam[i] = gam[i+1]*alp[i] + bet[i]
            
            # FORTRAN: Second elimination (lines 97-104)
            # In Fortran: alp(1) = 0.0, bet(1) = ebeg - 2.5*5.0*fbeg
            # Direct mapping to Python: alp[1] = 0.0, bet[1] = ...
            alp[1] = 0.0
            bet[1] = ebeg - 2.5*5.0*fbeg
            
            # do i=1,n-2
            for i in range(1, n-1):
                ip1 = i + 1
                # FORTRAN: alp(ip1) and bet(ip1) where ip1=i+1
                # Direct mapping: when i=1, ip1=2, set alp[2]
                alp[ip1] = -1.0 / (rhom + alp[i])
                bet[ip1] = alp[ip1] * (bet[i] - gam[i])
            
            # FORTRAN: Final coefficients (lines 106-121)
            e = np.zeros(n, dtype=np.float64)
            f = np.zeros(n, dtype=np.float64)
            d = np.zeros(n, dtype=np.float64)
            c = np.zeros(n, dtype=np.float64)
            b = np.zeros(n, dtype=np.float64)
            
            e[n-1] = eend + 2.5*5.0*fend
            e[n-2] = e[n-1]*alp[n-2] + bet[n-2]
            f[n-2] = (e[n-1] - e[n-2]) / 5.0
            e[n-3] = e[n-2]*alp[n-3] + bet[n-3]
            f[n-3] = (e[n-2] - e[n-3]) / 5.0
            # FORTRAN: d(n-2)=dend+... (line 111)
            d[n-2] = dend + 1.5*4.0*eend + 1.5**2*10.0*fend
            
            # do i=n-3,1,-1
            for i in range(n-3, 0, -1):
                # FORTRAN: e(i) = e(i+1)*alp(i) + bet(i)
                # Direct mapping: when i=3, calculate e[3] using e[4]
                e[i] = e[i+1]*alp[i] + bet[i]
                f[i] = (e[i+1] - e[i]) / 5.0
                # FORTRAN: d(i)=(a(i+3)-3.d0*a(i+2)+3.d0*a(i+1)-a(i))/6.d0 ...
                # Only calculate d[i] when i+3 < n (avoid out-of-bounds)
                if i+3 < n:
                    d[i] = (coeff[i+3] - 3.0*coeff[i+2] + 3.0*coeff[i+1] - coeff[i])/6.0 - \
                           (e[i+3] + 27.0*e[i+2] + 93.0*e[i+1] + 59.0*e[i])/30.0
                    c[i] = 0.5*(coeff[i+2] + coeff[i]) - coeff[i+1] - 0.5*d[i+1] - 2.5*d[i] - \
                           0.1*(e[i+2] + 18.0*e[i+1] + 31.0*e[i])
                    b[i] = coeff[i+1] - coeff[i] - c[i] - d[i] - 0.2*(4.0*e[i] + e[i+1])
            
            # FORTRAN: Boundary handling (lines 123-129)
            # do i=n-3,n
            for i in range(n-3, n):
                # FORTRAN: b(i) = b(i-1) + ... and c(i) = c(i-1) + ...
                # Direct mapping: when i=n-3, calculate b[n-3] using b[n-4]
                b[i] = b[i-1] + 2.0*c[i-1] + 3.0*d[i-1] + 4.0*e[i-1] + 5.0*f[i-1]
                c[i] = c[i-1] + 3.0*d[i-1] + 6.0*e[i-1] + 10.0*f[i-1]
                d[i] = d[i-1] + 4.0*e[i-1] + 10.0*f[i-1]
                # if(i.ne.n) f(i)= a(i+1)-a(i)-b(i)-c(i)-d(i)-e(i)
                if i != n-1:  # i+1 must be < n to avoid out-of-bounds
                    f[i] = coeff[i+1] - coeff[i] - b[i] - c[i] - d[i] - e[i]
            f[n-1] = f[n-2]
            
            # FORTRAN: Scaling (lines 131-140)
            # In Fortran: b=b*fac means multiply ALL elements
            fac = 1.0 / h_step
            b = b * fac
            fac = fac / h_step
            c = c * fac
            fac = fac / h_step
            d = d * fac
            fac = fac / h_step
            e = e * fac
            fac = fac / h_step
            f = f * fac
            
            # Now copy to coefficient array
            for i in range(n):
                coeff[n + i] = b[i]
                coeff[2*n + i] = c[i]
                coeff[3*n + i] = d[i]
                coeff[4*n + i] = e[i]
                coeff[5*n + i] = f[i]
            
        else:
            # Periodic quintic spline - EXACT 1:1 Fortran port from spl_five_per
            # From spl_three_to_five.f90 lines 147-260
            
            # Fortran constants
            rhop = 13.0 + np.sqrt(105.0)
            rhom = 13.0 - np.sqrt(105.0)
            
            # Working arrays - need size n+1 for periodic algorithm
            alp = np.zeros(n+1, dtype=np.float64)
            bet = np.zeros(n+1, dtype=np.float64)
            gam = np.zeros(n+1, dtype=np.float64)
            
            # FORTRAN: First elimination (lines 163-181)
            alp[1] = 0.0
            bet[1] = 0.0
            
            # do i=1,n-4
            for i in range(1, n-3):
                ip1 = i + 1
                alp[ip1] = -1.0 / (rhop + alp[i])
                bet[ip1] = alp[ip1] * (bet[i] - 5.0*(coeff[i+3] - 4.0*coeff[i+2] + 6.0*coeff[i+1] - 4.0*coeff[i] + coeff[i-1]))
            
            # Special periodic boundary terms
            alp[n-2] = -1.0 / (rhop + alp[n-3])
            bet[n-2] = alp[n-2] * (bet[n-3] - 5.0*(coeff[1] - 4.0*coeff[0] + 6.0*coeff[n-2] - 4.0*coeff[n-3] + coeff[n-4]))
            
            alp[n-1] = -1.0 / (rhop + alp[n-2])
            bet[n-1] = alp[n-1] * (bet[n-2] - 5.0*(coeff[2] - 4.0*coeff[1] + 6.0*coeff[0] - 4.0*coeff[n-2] + coeff[n-3]))
            
            alp[n] = -1.0 / (rhop + alp[n-1])
            bet[n] = alp[n] * (bet[n-1] - 5.0*(coeff[3] - 4.0*coeff[2] + 6.0*coeff[1] - 4.0*coeff[0] + coeff[n-2]))
            
            # FORTRAN: Back substitution (lines 182-185)
            gam[n] = bet[n]
            for i in range(n-1, 0, -1):
                gam[i] = gam[i+1]*alp[i] + bet[i]
            
            # FORTRAN: Sherman-Morrison correction (lines 187-195)
            xplu = np.sqrt(0.25*rhop**2 - 1.0) - 0.5*rhop
            xmin = -np.sqrt(0.25*rhop**2 - 1.0) - 0.5*rhop
            dummy = (1.0/xmin)**(n-1)
            gammao_m_redef = (gam[2] + xplu*gam[n]) / (1.0 - dummy) / (xmin - xplu)
            gammao_p = (gam[2] + xmin*gam[n]) / (xplu**(n-1) - 1.0) / (xplu - xmin)
            gam[1] = gam[1] + gammao_m_redef*dummy + gammao_p
            for i in range(2, n+1):
                gam[i] = gam[i] + gammao_m_redef*(1.0/xmin)**(n-i) + gammao_p*xplu**(i-1)
            
            # FORTRAN: Second elimination (lines 197-204)
            alp[1] = 0.0
            bet[1] = 0.0
            
            for i in range(1, n):
                ip1 = i + 1
                alp[ip1] = -1.0 / (rhom + alp[i])
                bet[ip1] = alp[ip1] * (bet[i] - gam[i])
            
            # FORTRAN: e coefficients (lines 206-219)
            e = np.zeros(n+1, dtype=np.float64)
            e[n] = bet[n]
            for i in range(n-1, 0, -1):
                e[i] = e[i+1]*alp[i] + bet[i]
            
            # Second Sherman-Morrison correction
            xplu = np.sqrt(0.25*rhom**2 - 1.0) - 0.5*rhom
            xmin = -np.sqrt(0.25*rhom**2 - 1.0) - 0.5*rhom
            dummy = (1.0/xmin)**(n-1)
            gammao_m_redef = (e[2] + xplu*e[n]) / (1.0 - dummy) / (xmin - xplu)
            gammao_p = (e[2] + xmin*e[n]) / (xplu**(n-1) - 1.0) / (xplu - xmin)
            e[1] = e[1] + gammao_m_redef*dummy + gammao_p
            for i in range(2, n+1):
                e[i] = e[i] + gammao_m_redef*(1.0/xmin)**(n-i) + gammao_p*xplu**(i-1)
            
            # FORTRAN: f coefficients (lines 221-224)
            f = np.zeros(n+1, dtype=np.float64)
            for i in range(n-1, 0, -1):
                f[i] = (e[i+1] - e[i]) / 5.0
            f[n] = f[1]
            
            # FORTRAN: d coefficients (lines 226-234)
            d = np.zeros(n+1, dtype=np.float64)
            # d(n-1) in Fortran is d[n-2] in Python
            d[n-2] = (coeff[2] - 3.0*coeff[1] + 3.0*coeff[0] - coeff[n-2])/6.0 - \
                     (e[3] + 27.0*e[2] + 93.0*e[1] + 59.0*e[n-2])/30.0
            # d(n-2) in Fortran is d[n-3] in Python  
            d[n-3] = (coeff[1] - 3.0*coeff[0] + 3.0*coeff[n-2] - coeff[n-3])/6.0 - \
                     (e[2] + 27.0*e[1] + 93.0*e[n-2] + 59.0*e[n-3])/30.0
            # do i=n-3,1,-1 (Fortran) means d(n-3) down to d(1)
            # In Python: d[n-4] down to d[0]
            for i in range(n-4, -1, -1):
                d[i] = (coeff[i+2] - 3.0*coeff[i+1] + 3.0*coeff[i] - coeff[i-1])/6.0 - \
                       (e[i+3] + 27.0*e[i+2] + 93.0*e[i+1] + 59.0*e[i])/30.0
            d[n] = d[1]
            
            # FORTRAN: c and b coefficients (lines 235-245)
            c = np.zeros(n+1, dtype=np.float64)
            b = np.zeros(n+1, dtype=np.float64)
            c[n-2] = 0.5*(coeff[1] + coeff[n-2]) - coeff[0] - 0.5*d[1] - 2.5*d[n-2] - \
                     0.1*(e[2] + 18.0*e[1] + 31.0*e[n-2])
            b[n-2] = coeff[0] - coeff[n-2] - c[n-2] - d[n-2] - 0.2*(4.0*e[n-2] + e[1])
            
            for i in range(n-3, 0, -1):
                c[i] = 0.5*(coeff[i+1] + coeff[i-1]) - coeff[i] - 0.5*d[i] - 2.5*d[i-1] - \
                       0.1*(e[i+2] + 18.0*e[i+1] + 31.0*e[i])
                b[i] = coeff[i] - coeff[i-1] - c[i] - d[i] - 0.2*(4.0*e[i] + e[i+1])
            b[n] = b[1]
            c[n] = c[1]
            
            # FORTRAN: Scaling (lines 247-256)
            fac = 1.0 / h_step
            b = b * fac
            fac = fac / h_step
            c = c * fac
            fac = fac / h_step
            d = d * fac
            fac = fac / h_step
            e = e * fac
            fac = fac / h_step
            f = f * fac
            
            # Now copy to coefficient array
            for i in range(n):
                coeff[n + i] = b[i]
                coeff[2*n + i] = c[i]
                coeff[3*n + i] = d[i]
                coeff[4*n + i] = e[i]
                coeff[5*n + i] = f[i]


# ==== 1D SPLINE EVALUATION ====
@cfunc(types.void(
    types.int32,                        # order
    types.int32,                        # num_points
    types.int32,                        # periodic (0 or 1)
    types.float64,                      # x_min
    types.float64,                      # h_step
    types.CPointer(types.float64),      # coeff array
    types.float64,                      # x
    types.CPointer(types.float64)       # output y
), nopython=True, nogil=True, cache=True, fastmath=True)
def evaluate_splines_1d_cfunc(order, num_points, periodic, x_min, h_step, coeff, x, y_out):
    """Evaluate 1D spline at point x"""
    
    # Find interval
    xj = x
    if periodic:
        period = h_step * num_points
        # Manual modulo for periodic boundary
        xj = x - x_min
        if xj < 0:
            n_periods = int((-xj / period) + 1)
            xj = xj + period * n_periods
        elif xj >= period:
            n_periods = int(xj / period)
            xj = xj - period * n_periods
        xj = xj + x_min
    
    x_norm = (xj - x_min) / h_step
    interval_index = int(x_norm)
    
    # Handle boundary conditions
    if periodic:
        # For periodic splines, wrap around
        interval_index = interval_index % num_points
    else:
        # For non-periodic splines, clamp to valid range
        if interval_index < 0:
            interval_index = 0
        elif interval_index >= num_points - 1:
            interval_index = num_points - 2
    
    # Local coordinate within interval
    x_local = (x_norm - interval_index) * h_step
    
    # Evaluate polynomial using Horner's method
    if order == 3:
        # Cubic: y = a + b*x + c*x^2 + d*x^3
        a = coeff[interval_index]
        b = coeff[num_points + interval_index]
        c = coeff[2*num_points + interval_index]
        d = coeff[3*num_points + interval_index]
        y_out[0] = a + x_local * (b + x_local * (c + x_local * d))
    elif order == 4:
        # Quartic
        a = coeff[interval_index]
        b = coeff[num_points + interval_index]
        c = coeff[2*num_points + interval_index]
        d = coeff[3*num_points + interval_index]
        e = coeff[4*num_points + interval_index]
        y_out[0] = a + x_local * (b + x_local * (c + x_local * (d + x_local * e)))
    elif order == 5:
        # Quintic
        a = coeff[interval_index]
        b = coeff[num_points + interval_index]
        c = coeff[2*num_points + interval_index]
        d = coeff[3*num_points + interval_index]
        e = coeff[4*num_points + interval_index]
        f = coeff[5*num_points + interval_index]
        y_out[0] = a + x_local * (b + x_local * (c + x_local * (d + x_local * (e + x_local * f))))
    else:
        # General case
        y = coeff[order * num_points + interval_index]
        for k_power in range(order - 1, -1, -1):
            y = coeff[k_power * num_points + interval_index] + x_local * y
        y_out[0] = y


# ==== 2D SPLINE CONSTRUCTION ====
@cfunc(types.void(
    types.CPointer(types.float64),      # x_min array (2)
    types.CPointer(types.float64),      # x_max array (2)
    types.CPointer(types.float64),      # y values (flattened, n1 x n2)
    types.CPointer(types.int32),        # num_points array (2)
    types.CPointer(types.int32),        # order array (2)
    types.CPointer(types.int32),        # periodic array (2)
    types.CPointer(types.float64),      # output coeff array (flattened)
    types.CPointer(types.float64),      # workspace array for temp_y (size >= max(n1,n2))
    types.CPointer(types.float64),      # workspace array for temp_coeff (size >= 6*max(n1,n2))
), nopython=True, nogil=True, cache=True, fastmath=True)
def construct_splines_2d_cfunc(x_min, x_max, y, num_points, order, periodic, coeff, workspace_y, workspace_coeff):
    """Construct 2D spline using tensor product approach"""
    
    # Extract dimensions
    n1 = num_points[0]
    n2 = num_points[1]
    o1 = order[0]
    o2 = order[1]
    
    # Calculate h_step for both dimensions
    h1 = (x_max[0] - x_min[0]) / (n1 - 1)
    h2 = (x_max[1] - x_min[1]) / (n2 - 1)
    
    # Initialize coefficient array to zero
    for i in range((o1+1)*(o2+1)*n1*n2):
        coeff[i] = 0.0
    
    # Step 1: Apply 1D splines along dimension 2 (for each row)
    # This matches Fortran: spl%coeff(0, :, i1, :) = splcoe
    for i1 in range(n1):
        # Extract row into workspace
        for i2 in range(n2):
            workspace_y[i2] = y[i1*n2 + i2]
        
        # Construct 1D spline for this row
        construct_splines_1d_cfunc(x_min[1], x_max[1], workspace_y, n2, o2, periodic[1], workspace_coeff)
        
        # Copy coefficients back - store at coeff(0, k2, i1, i2)
        # Fortran: spl%coeff(0, :, i1, :) = splcoe
        # Python: coeff[k1*(o2+1)*n1*n2 + k2*n1*n2 + i1*n2 + i2] with k1=0
        for k2 in range(o2 + 1):
            for i2 in range(n2):
                idx = 0*(o2+1)*n1*n2 + k2*n1*n2 + i1*n2 + i2
                coeff[idx] = workspace_coeff[k2*n2 + i2]
    
    # Step 2: Apply 1D splines along dimension 1 (for each column and coefficient)
    # This matches Fortran: splcoe(0,:) = spl%coeff(0, k2, :, i2)
    #                      spl%coeff(:, k2, :, i2) = splcoe
    for i2 in range(n2):
        for k2 in range(o2 + 1):
            # Extract column into workspace - get coeff(0, k2, :, i2)
            # Fortran: splcoe(0,:) = spl%coeff(0, k2, :, i2)
            for i1 in range(n1):
                idx = 0*(o2+1)*n1*n2 + k2*n1*n2 + i1*n2 + i2
                workspace_y[i1] = coeff[idx]
            
            # Construct 1D spline for this column
            construct_splines_1d_cfunc(x_min[0], x_max[0], workspace_y, n1, o1, periodic[0], workspace_coeff)
            
            # Copy coefficients back - store at coeff(k1, k2, i1, i2)
            # Fortran: spl%coeff(:, k2, :, i2) = splcoe
            for k1 in range(o1 + 1):
                for i1 in range(n1):
                    idx = k1*(o2+1)*n1*n2 + k2*n1*n2 + i1*n2 + i2
                    coeff[idx] = workspace_coeff[k1*n1 + i1]


# ==== 2D SPLINE EVALUATION ====
@cfunc(types.void(
    types.CPointer(types.int32),        # order array (2)
    types.CPointer(types.int32),        # num_points array (2)
    types.CPointer(types.int32),        # periodic array (2)
    types.CPointer(types.float64),      # x_min array (2)
    types.CPointer(types.float64),      # h_step array (2)
    types.CPointer(types.float64),      # coeff array (flattened)
    types.CPointer(types.float64),      # x array (2)
    types.CPointer(types.float64)       # output y
), nopython=True, nogil=True, cache=True, fastmath=True)
def evaluate_splines_2d_cfunc(order, num_points, periodic, x_min, h_step, coeff, x, y_out):
    """Evaluate 2D spline at point (x[0], x[1])"""
    
    # Extract parameters
    o1 = order[0]
    o2 = order[1]
    n1 = num_points[0]
    n2 = num_points[1]
    
    # Find intervals for both dimensions
    # Dimension 1
    xj1 = x[0]
    if periodic[0]:
        period1 = h_step[0] * (n1 - 1)
        xj1 = x[0] - x_min[0]
        if xj1 < 0:
            n_periods = int((-xj1 / period1) + 1)
            xj1 = xj1 + period1 * n_periods
        elif xj1 >= period1:
            n_periods = int(xj1 / period1)
            xj1 = xj1 - period1 * n_periods
        xj1 = xj1 + x_min[0]
    
    x_norm_1 = (xj1 - x_min[0]) / h_step[0]
    interval_1 = int(x_norm_1)
    if interval_1 < 0:
        interval_1 = 0
    elif interval_1 >= n1 - 1:
        interval_1 = n1 - 2
    x_local_1 = (x_norm_1 - interval_1) * h_step[0]
    
    # Dimension 2
    xj2 = x[1]
    if periodic[1]:
        period2 = h_step[1] * (n2 - 1)
        xj2 = x[1] - x_min[1]
        if xj2 < 0:
            n_periods = int((-xj2 / period2) + 1)
            xj2 = xj2 + period2 * n_periods
        elif xj2 >= period2:
            n_periods = int(xj2 / period2)
            xj2 = xj2 - period2 * n_periods
        xj2 = xj2 + x_min[1]
    
    x_norm_2 = (xj2 - x_min[1]) / h_step[1]
    interval_2 = int(x_norm_2)
    if interval_2 < 0:
        interval_2 = 0
    elif interval_2 >= n2 - 1:
        interval_2 = n2 - 2
    x_local_2 = (x_norm_2 - interval_2) * h_step[1]
    
    # Extract coefficients for this interval - matching Fortran approach exactly
    # Fortran: coeff_local(:,:) = spl%coeff(:, :, interval_index(1) + 1, interval_index(2) + 1)
    
    # Create coeff_2 array by evaluating along dimension 1 first
    # Fortran: coeff_2(:) = coeff_local(spl%order(1), 0:spl%order(2))
    coeff_2 = [0.0] * (o2 + 1)
    for k2 in range(o2 + 1):
        # Start with highest order k1 = o1
        idx = o1*(o2+1)*n1*n2 + k2*n1*n2 + interval_1*n2 + interval_2
        coeff_2[k2] = coeff[idx]
        
        # Evaluate polynomial in x_local_1 using Horner's method
        # Fortran: coeff_2(:) = coeff_local(k1, :) + x_local(1)*coeff_2
        for k1 in range(o1 - 1, -1, -1):
            idx = k1*(o2+1)*n1*n2 + k2*n1*n2 + interval_1*n2 + interval_2
            coeff_2[k2] = coeff[idx] + x_local_1 * coeff_2[k2]
    
    # Now evaluate along dimension 2 using coeff_2
    # Fortran: y = coeff_2(spl%order(2))
    # Fortran: do k2 = spl%order(2)-1, 0, -1
    # Fortran:     y = coeff_2(k2) + x_local(2)*y
    # Fortran: enddo
    y = coeff_2[o2]
    for k2 in range(o2 - 1, -1, -1):
        y = coeff_2[k2] + x_local_2 * y
    
    y_out[0] = y


# ==== 1D DERIVATIVE EVALUATION ====
@cfunc(types.void(
    types.int32,                        # order
    types.int32,                        # num_points
    types.int32,                        # periodic (0 or 1)
    types.float64,                      # x_min
    types.float64,                      # h_step
    types.CPointer(types.float64),      # coeff array
    types.float64,                      # x
    types.CPointer(types.float64),      # output y
    types.CPointer(types.float64),      # output dy
), nopython=True, nogil=True, cache=True, fastmath=True)
def evaluate_splines_1d_der_cfunc(order, num_points, periodic, x_min, h_step, coeff, x, y_out, dy_out):
    """Evaluate 1D spline and its derivative at point x"""
    
    # Find interval (same as evaluation)
    xj = x
    if periodic:
        period = h_step * num_points
        xj = x - x_min
        if xj < 0:
            n_periods = int((-xj / period) + 1)
            xj = xj + period * n_periods
        elif xj >= period:
            n_periods = int(xj / period)
            xj = xj - period * n_periods
        xj = xj + x_min
    
    x_norm = (xj - x_min) / h_step
    interval_index = int(x_norm)
    
    if interval_index < 0:
        interval_index = 0
    elif interval_index >= num_points - 1:
        interval_index = num_points - 2
    
    x_local = (x_norm - interval_index) * h_step
    
    # Evaluate value and derivative using Horner's method
    if order == 3:
        a = coeff[interval_index]
        b = coeff[num_points + interval_index]
        c = coeff[2*num_points + interval_index]
        d = coeff[3*num_points + interval_index]
        
        y_out[0] = a + x_local * (b + x_local * (c + x_local * d))
        dy_out[0] = b + x_local * (2.0 * c + x_local * 3.0 * d)
    else:
        # General case
        y = coeff[order * num_points + interval_index]
        dy = order * coeff[order * num_points + interval_index]
        
        for k in range(order - 1, 0, -1):
            y = coeff[k * num_points + interval_index] + x_local * y
            dy = k * coeff[k * num_points + interval_index] + x_local * dy
        
        y_out[0] = coeff[interval_index] + x_local * y
        dy_out[0] = dy


# ==== 1D SECOND DERIVATIVE EVALUATION ====
@cfunc(types.void(
    types.int32,                        # order
    types.int32,                        # num_points
    types.int32,                        # periodic (0 or 1)
    types.float64,                      # x_min
    types.float64,                      # h_step
    types.CPointer(types.float64),      # coeff array
    types.float64,                      # x
    types.CPointer(types.float64),      # output y
    types.CPointer(types.float64),      # output dy
    types.CPointer(types.float64),      # output d2y
), nopython=True, nogil=True, cache=True, fastmath=True)
def evaluate_splines_1d_der2_cfunc(order, num_points, periodic, x_min, h_step, coeff, x, y_out, dy_out, d2y_out):
    """Evaluate 1D spline and its first and second derivatives at point x"""
    
    # Find interval (same as evaluation)
    xj = x
    if periodic:
        period = h_step * num_points
        xj = x - x_min
        if xj < 0:
            n_periods = int((-xj / period) + 1)
            xj = xj + period * n_periods
        elif xj >= period:
            n_periods = int(xj / period)
            xj = xj - period * n_periods
        xj = xj + x_min
    
    x_norm = (xj - x_min) / h_step
    interval_index = int(x_norm)
    
    if interval_index < 0:
        interval_index = 0
    elif interval_index >= num_points - 1:
        interval_index = num_points - 2
    
    x_local = (x_norm - interval_index) * h_step
    
    # Evaluate value and derivatives using Horner's method
    if order == 3:
        a = coeff[interval_index]
        b = coeff[num_points + interval_index]
        c = coeff[2*num_points + interval_index]
        d = coeff[3*num_points + interval_index]
        
        y_out[0] = a + x_local * (b + x_local * (c + x_local * d))
        dy_out[0] = b + x_local * (2.0 * c + x_local * 3.0 * d)
        d2y_out[0] = 2.0 * c + x_local * 6.0 * d
    else:
        # General case - value
        y = coeff[order * num_points + interval_index]
        for k in range(order - 1, -1, -1):
            y = coeff[k * num_points + interval_index] + x_local * y
        y_out[0] = y
        
        # First derivative
        dy = order * coeff[order * num_points + interval_index]
        for k in range(order - 1, 0, -1):
            dy = k * coeff[k * num_points + interval_index] + x_local * dy
        dy_out[0] = dy
        
        # Second derivative
        d2y = order * (order - 1) * coeff[order * num_points + interval_index]
        for k in range(order - 1, 1, -1):
            d2y = k * (k - 1) * coeff[k * num_points + interval_index] + x_local * d2y
        d2y_out[0] = d2y


# ==== 2D SPLINE DERIVATIVE EVALUATION ====
@cfunc(types.void(
    types.CPointer(types.int32),        # order array (2)
    types.CPointer(types.int32),        # num_points array (2)
    types.CPointer(types.int32),        # periodic array (2)
    types.CPointer(types.float64),      # x_min array (2)
    types.CPointer(types.float64),      # h_step array (2)
    types.CPointer(types.float64),      # coeff array (flattened)
    types.CPointer(types.float64),      # x array (2)
    types.CPointer(types.float64),      # output y
    types.CPointer(types.float64),      # output dy/dx1
    types.CPointer(types.float64),      # output dy/dx2
), nopython=True, nogil=True, cache=True, fastmath=True)
def evaluate_splines_2d_der_cfunc(order, num_points, periodic, x_min, h_step, coeff, x, y_out, dydx1_out, dydx2_out):
    """Evaluate 2D spline and its first derivatives at point (x[0], x[1])"""
    
    # Extract parameters
    o1 = order[0]
    o2 = order[1]
    n1 = num_points[0]
    n2 = num_points[1]
    
    # Find intervals for both dimensions
    # Dimension 1
    xj1 = x[0]
    if periodic[0]:
        period1 = h_step[0] * (n1 - 1)
        xj1 = x[0] - x_min[0]
        if xj1 < 0:
            n_periods = int((-xj1 / period1) + 1)
            xj1 = xj1 + period1 * n_periods
        elif xj1 >= period1:
            n_periods = int(xj1 / period1)
            xj1 = xj1 - period1 * n_periods
        xj1 = xj1 + x_min[0]
    
    x_norm_1 = (xj1 - x_min[0]) / h_step[0]
    interval_1 = int(x_norm_1)
    if interval_1 < 0:
        interval_1 = 0
    elif interval_1 >= n1 - 1:
        interval_1 = n1 - 2
    x_local_1 = (x_norm_1 - interval_1) * h_step[0]
    
    # Dimension 2
    xj2 = x[1]
    if periodic[1]:
        period2 = h_step[1] * (n2 - 1)
        xj2 = x[1] - x_min[1]
        if xj2 < 0:
            n_periods = int((-xj2 / period2) + 1)
            xj2 = xj2 + period2 * n_periods
        elif xj2 >= period2:
            n_periods = int(xj2 / period2)
            xj2 = xj2 - period2 * n_periods
        xj2 = xj2 + x_min[1]
    
    x_norm_2 = (xj2 - x_min[1]) / h_step[1]
    interval_2 = int(x_norm_2)
    if interval_2 < 0:
        interval_2 = 0
    elif interval_2 >= n2 - 1:
        interval_2 = n2 - 2
    x_local_2 = (x_norm_2 - interval_2) * h_step[1]
    
    # Evaluate using tensor product with derivatives
    # We need to evaluate:
    # - Value: sum_k1 sum_k2 c_k1,k2 * x1^k1 * x2^k2
    # - dy/dx1: sum_k1 sum_k2 k1 * c_k1,k2 * x1^(k1-1) * x2^k2
    # - dy/dx2: sum_k1 sum_k2 k2 * c_k1,k2 * x1^k1 * x2^(k2-1)
    
    y = 0.0
    dydx1 = 0.0
    dydx2 = 0.0
    
    # Evaluate polynomials using Horner's method
    # First for value
    for k1 in range(o1, -1, -1):
        base_idx = k1*(o2+1)*n1*n2 + interval_1*n2 + interval_2
        
        # Evaluate in x2 direction
        y_temp = coeff[base_idx + o2*n1*n2]
        for k2 in range(o2 - 1, -1, -1):
            y_temp = coeff[base_idx + k2*n1*n2] + x_local_2 * y_temp
        
        if k1 == o1:
            y = y_temp
        else:
            y = y_temp + x_local_1 * y
    
    # For dy/dx1
    for k1 in range(o1, 0, -1):
        base_idx = k1*(o2+1)*n1*n2 + interval_1*n2 + interval_2
        
        # Evaluate in x2 direction
        y_temp = coeff[base_idx + o2*n1*n2]
        for k2 in range(o2 - 1, -1, -1):
            y_temp = coeff[base_idx + k2*n1*n2] + x_local_2 * y_temp
        
        if k1 == o1:
            dydx1 = k1 * y_temp
        else:
            dydx1 = k1 * y_temp + x_local_1 * dydx1
    
    # For dy/dx2
    for k1 in range(o1, -1, -1):
        base_idx = k1*(o2+1)*n1*n2 + interval_1*n2 + interval_2
        
        # Evaluate derivative in x2 direction
        dy_temp = o2 * coeff[base_idx + o2*n1*n2]
        for k2 in range(o2 - 1, 0, -1):
            dy_temp = k2 * coeff[base_idx + k2*n1*n2] + x_local_2 * dy_temp
        
        if k1 == o1:
            dydx2 = dy_temp
        else:
            dydx2 = dy_temp + x_local_1 * dydx2
    
    y_out[0] = y
    dydx1_out[0] = dydx1
    dydx2_out[0] = dydx2


# ==== 3D SPLINE CONSTRUCTION ====
@cfunc(types.void(
    types.CPointer(types.float64),      # x_min array (3)
    types.CPointer(types.float64),      # x_max array (3)
    types.CPointer(types.float64),      # y values (flattened, n1 x n2 x n3)
    types.CPointer(types.int32),        # num_points array (3)
    types.CPointer(types.int32),        # order array (3)
    types.CPointer(types.int32),        # periodic array (3)
    types.CPointer(types.float64),      # output coeff array (flattened)
    types.CPointer(types.float64),      # workspace array for 1D data
    types.CPointer(types.float64),      # workspace array for 1D coeffs
    types.CPointer(types.float64),      # workspace array for 2D construction
), nopython=True, nogil=True, cache=True, fastmath=True)
def construct_splines_3d_cfunc(x_min, x_max, y, num_points, order, periodic, coeff, work_1d, work_1d_coeff, work_2d_coeff):
    """Construct 3D spline using tensor product approach"""
    
    # Extract dimensions
    n1 = num_points[0]
    n2 = num_points[1]
    n3 = num_points[2]
    o1 = order[0]
    o2 = order[1]
    o3 = order[2]
    
    # Calculate h_step for all dimensions
    h1 = (x_max[0] - x_min[0]) / (n1 - 1)
    h2 = (x_max[1] - x_min[1]) / (n2 - 1)
    h3 = (x_max[2] - x_min[2]) / (n3 - 1)
    
    # Total size for intermediate storage
    n12 = n1 * n2
    n123 = n1 * n2 * n3
    
    # Step 1: Apply 1D splines along dimension 3 (for each (i1,i2) line)
    for i1 in range(n1):
        for i2 in range(n2):
            # Extract line along dimension 3
            for i3 in range(n3):
                work_1d[i3] = y[i1*n2*n3 + i2*n3 + i3]
            
            # Construct 1D spline for this line
            construct_splines_1d_cfunc(x_min[2], x_max[2], work_1d, n3, o3, periodic[2], work_1d_coeff)
            
            # Copy coefficients back
            for k3 in range(o3 + 1):
                for i3 in range(n3):
                    coeff[k3*n123 + i1*n2*n3 + i2*n3 + i3] = work_1d_coeff[k3*n3 + i3]
    
    # Step 2: Apply 1D splines along dimension 2 (for each (i1,k3) and each i3)
    for k3 in range(o3 + 1):
        for i1 in range(n1):
            for i3 in range(n3):
                # Extract line along dimension 2
                for i2 in range(n2):
                    work_1d[i2] = coeff[k3*n123 + i1*n2*n3 + i2*n3 + i3]
                
                # Construct 1D spline for this line
                construct_splines_1d_cfunc(x_min[1], x_max[1], work_1d, n2, o2, periodic[1], work_1d_coeff)
                
                # Store in temporary 2D coefficient array
                for k2 in range(o2 + 1):
                    for i2 in range(n2):
                        work_2d_coeff[k2*(o3+1)*n123 + k3*n123 + i1*n2*n3 + i2*n3 + i3] = work_1d_coeff[k2*n2 + i2]
    
    # Step 3: Apply 1D splines along dimension 1 (for each (k2,k3) and each (i2,i3))
    for k2 in range(o2 + 1):
        for k3 in range(o3 + 1):
            for i2 in range(n2):
                for i3 in range(n3):
                    # Extract line along dimension 1
                    for i1 in range(n1):
                        work_1d[i1] = work_2d_coeff[k2*(o3+1)*n123 + k3*n123 + i1*n2*n3 + i2*n3 + i3]
                    
                    # Construct 1D spline for this line
                    construct_splines_1d_cfunc(x_min[0], x_max[0], work_1d, n1, o1, periodic[0], work_1d_coeff)
                    
                    # Copy final coefficients
                    for k1 in range(o1 + 1):
                        for i1 in range(n1):
                            idx = k1*(o2+1)*(o3+1)*n123 + k2*(o3+1)*n123 + k3*n123 + i1*n2*n3 + i2*n3 + i3
                            coeff[idx] = work_1d_coeff[k1*n1 + i1]


# ==== 3D SPLINE EVALUATION ====
@cfunc(types.void(
    types.CPointer(types.int32),        # order array (3)
    types.CPointer(types.int32),        # num_points array (3)
    types.CPointer(types.int32),        # periodic array (3)
    types.CPointer(types.float64),      # x_min array (3)
    types.CPointer(types.float64),      # h_step array (3)
    types.CPointer(types.float64),      # coeff array (flattened)
    types.CPointer(types.float64),      # x array (3)
    types.CPointer(types.float64)       # output y
), nopython=True, nogil=True, cache=True, fastmath=True)
def evaluate_splines_3d_cfunc(order, num_points, periodic, x_min, h_step, coeff, x, y_out):
    """Evaluate 3D spline at point (x[0], x[1], x[2])"""
    
    # Extract parameters
    o1 = order[0]
    o2 = order[1]
    o3 = order[2]
    n1 = num_points[0]
    n2 = num_points[1]
    n3 = num_points[2]
    n123 = n1 * n2 * n3
    
    # Find intervals for all dimensions
    intervals = [0, 0, 0]
    x_locals = [0.0, 0.0, 0.0]
    
    # Process each dimension
    for dim in range(3):
        xj = x[dim]
        if periodic[dim]:
            period = h_step[dim] * (num_points[dim] - 1)
            xj = x[dim] - x_min[dim]
            if xj < 0:
                n_periods = int((-xj / period) + 1)
                xj = xj + period * n_periods
            elif xj >= period:
                n_periods = int(xj / period)
                xj = xj - period * n_periods
            xj = xj + x_min[dim]
        
        x_norm = (xj - x_min[dim]) / h_step[dim]
        interval = int(x_norm)
        if interval < 0:
            interval = 0
        elif interval >= num_points[dim] - 1:
            interval = num_points[dim] - 2
        
        intervals[dim] = interval
        x_locals[dim] = (x_norm - interval) * h_step[dim]
    
    # Evaluate using tensor product
    # We evaluate polynomial in order: dimension 3, then 2, then 1
    y = 0.0
    
    # Triple nested Horner's method
    for k1 in range(o1, -1, -1):
        # For this k1, evaluate 2D polynomial in (x2, x3)
        y2d = 0.0
        
        for k2 in range(o2, -1, -1):
            # For this (k1, k2), evaluate 1D polynomial in x3
            idx_base = k1*(o2+1)*(o3+1)*n123 + k2*(o3+1)*n123 + intervals[0]*n2*n3 + intervals[1]*n3 + intervals[2]
            
            y1d = coeff[idx_base + o3*n123]
            for k3 in range(o3 - 1, -1, -1):
                y1d = coeff[idx_base + k3*n123] + x_locals[2] * y1d
            
            if k2 == o2:
                y2d = y1d
            else:
                y2d = y1d + x_locals[1] * y2d
        
        if k1 == o1:
            y = y2d
        else:
            y = y2d + x_locals[0] * y
    
    y_out[0] = y


# ==== 3D SPLINE DERIVATIVE EVALUATION ====
@cfunc(types.void(
    types.CPointer(types.int32),        # order array (3)
    types.CPointer(types.int32),        # num_points array (3)
    types.CPointer(types.int32),        # periodic array (3)
    types.CPointer(types.float64),      # x_min array (3)
    types.CPointer(types.float64),      # h_step array (3)
    types.CPointer(types.float64),      # coeff array (flattened)
    types.CPointer(types.float64),      # x array (3)
    types.CPointer(types.float64),      # output y
    types.CPointer(types.float64),      # output dy/dx1
    types.CPointer(types.float64),      # output dy/dx2
    types.CPointer(types.float64),      # output dy/dx3
), nopython=True, nogil=True, cache=True, fastmath=True)
def evaluate_splines_3d_der_cfunc(order, num_points, periodic, x_min, h_step, coeff, x, y_out, dydx1_out, dydx2_out, dydx3_out):
    """Evaluate 3D spline and its first derivatives at point (x[0], x[1], x[2])"""
    
    # Extract parameters
    o1 = order[0]
    o2 = order[1]
    o3 = order[2]
    n1 = num_points[0]
    n2 = num_points[1]
    n3 = num_points[2]
    n123 = n1 * n2 * n3
    
    # Find intervals for all dimensions
    intervals = [0, 0, 0]
    x_locals = [0.0, 0.0, 0.0]
    
    # Process each dimension
    for dim in range(3):
        xj = x[dim]
        if periodic[dim]:
            period = h_step[dim] * (num_points[dim] - 1)
            xj = x[dim] - x_min[dim]
            if xj < 0:
                n_periods = int((-xj / period) + 1)
                xj = xj + period * n_periods
            elif xj >= period:
                n_periods = int(xj / period)
                xj = xj - period * n_periods
            xj = xj + x_min[dim]
        
        x_norm = (xj - x_min[dim]) / h_step[dim]
        interval = int(x_norm)
        if interval < 0:
            interval = 0
        elif interval >= num_points[dim] - 1:
            interval = num_points[dim] - 2
        
        intervals[dim] = interval
        x_locals[dim] = (x_norm - interval) * h_step[dim]
    
    # Evaluate value and all three partial derivatives
    y = 0.0
    dydx1 = 0.0
    dydx2 = 0.0
    dydx3 = 0.0
    
    # For the value
    for k1 in range(o1, -1, -1):
        y2d = 0.0
        for k2 in range(o2, -1, -1):
            idx_base = k1*(o2+1)*(o3+1)*n123 + k2*(o3+1)*n123 + intervals[0]*n2*n3 + intervals[1]*n3 + intervals[2]
            y1d = coeff[idx_base + o3*n123]
            for k3 in range(o3 - 1, -1, -1):
                y1d = coeff[idx_base + k3*n123] + x_locals[2] * y1d
            if k2 == o2:
                y2d = y1d
            else:
                y2d = y1d + x_locals[1] * y2d
        if k1 == o1:
            y = y2d
        else:
            y = y2d + x_locals[0] * y
    
    # For dy/dx1
    for k1 in range(o1, 0, -1):
        y2d = 0.0
        for k2 in range(o2, -1, -1):
            idx_base = k1*(o2+1)*(o3+1)*n123 + k2*(o3+1)*n123 + intervals[0]*n2*n3 + intervals[1]*n3 + intervals[2]
            y1d = coeff[idx_base + o3*n123]
            for k3 in range(o3 - 1, -1, -1):
                y1d = coeff[idx_base + k3*n123] + x_locals[2] * y1d
            if k2 == o2:
                y2d = y1d
            else:
                y2d = y1d + x_locals[1] * y2d
        if k1 == o1:
            dydx1 = k1 * y2d
        else:
            dydx1 = k1 * y2d + x_locals[0] * dydx1
    
    # For dy/dx2
    for k1 in range(o1, -1, -1):
        dy2d = 0.0
        for k2 in range(o2, 0, -1):
            idx_base = k1*(o2+1)*(o3+1)*n123 + k2*(o3+1)*n123 + intervals[0]*n2*n3 + intervals[1]*n3 + intervals[2]
            y1d = coeff[idx_base + o3*n123]
            for k3 in range(o3 - 1, -1, -1):
                y1d = coeff[idx_base + k3*n123] + x_locals[2] * y1d
            if k2 == o2:
                dy2d = k2 * y1d
            else:
                dy2d = k2 * y1d + x_locals[1] * dy2d
        if k1 == o1:
            dydx2 = dy2d
        else:
            dydx2 = dy2d + x_locals[0] * dydx2
    
    # For dy/dx3
    for k1 in range(o1, -1, -1):
        y2d = 0.0
        for k2 in range(o2, -1, -1):
            idx_base = k1*(o2+1)*(o3+1)*n123 + k2*(o3+1)*n123 + intervals[0]*n2*n3 + intervals[1]*n3 + intervals[2]
            dy1d = o3 * coeff[idx_base + o3*n123]
            for k3 in range(o3 - 1, 0, -1):
                dy1d = k3 * coeff[idx_base + k3*n123] + x_locals[2] * dy1d
            if k2 == o2:
                y2d = dy1d
            else:
                y2d = dy1d + x_locals[1] * y2d
        if k1 == o1:
            dydx3 = y2d
        else:
            dydx3 = y2d + x_locals[0] * dydx3
    
    y_out[0] = y
    dydx1_out[0] = dydx1
    dydx2_out[0] = dydx2
    dydx3_out[0] = dydx3


# Export cfunc addresses
def get_cfunc_addresses():
    """Return dictionary of cfunc addresses for ctypes access"""
    return {
        'construct_splines_1d': construct_splines_1d_cfunc.address,
        'evaluate_splines_1d': evaluate_splines_1d_cfunc.address,
        'evaluate_splines_1d_der': evaluate_splines_1d_der_cfunc.address,
        'evaluate_splines_1d_der2': evaluate_splines_1d_der2_cfunc.address,
        'construct_splines_2d': construct_splines_2d_cfunc.address,
        'evaluate_splines_2d': evaluate_splines_2d_cfunc.address,
        'evaluate_splines_2d_der': evaluate_splines_2d_der_cfunc.address,
        'construct_splines_3d': construct_splines_3d_cfunc.address,
        'evaluate_splines_3d': evaluate_splines_3d_cfunc.address,
        'evaluate_splines_3d_der': evaluate_splines_3d_der_cfunc.address,
    }