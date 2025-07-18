#!/usr/bin/env python3
"""
Complete rewrite of quintic implementation - exact Fortran port
"""

def generate_exact_quintic_implementation():
    """Generate the exact quintic implementation code"""
    
    quintic_code = '''
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
            alp[0] = 0.0
            bet[0] = ebeg*(2.0 + rhom) - 5.0*fbeg*(3.0 + 1.5*rhom)
            
            for i in range(n-4):
                ip1 = i + 1
                alp[ip1] = -1.0 / (rhop + alp[i])
                # FORTRAN: 5.d0*(a(i+4)-4.d0*a(i+3)+6.d0*a(i+2)-4.d0*a(ip1)+a(i))
                bet[ip1] = alp[ip1] * (bet[i] - 5.0*(coeff[i+4] - 4.0*coeff[i+3] + 6.0*coeff[i+2] - 4.0*coeff[ip1] + coeff[i]))
            
            # FORTRAN: Back substitution (lines 92-95)
            gam[n-3] = eend*(2.0 + rhom) + 5.0*fend*(3.0 + 1.5*rhom)
            for i in range(n-4, -1, -1):
                gam[i] = gam[i+1]*alp[i] + bet[i]
            
            # FORTRAN: Second elimination (lines 97-104)
            alp[0] = 0.0
            bet[0] = ebeg - 2.5*5.0*fbeg
            
            for i in range(n-2):
                ip1 = i + 1
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
            d[n-3] = dend + 1.5*4.0*eend + 1.5**2*10.0*fend
            
            for i in range(n-4, -1, -1):
                e[i] = e[i+1]*alp[i] + bet[i]
                f[i] = (e[i+1] - e[i]) / 5.0
                d[i] = (coeff[i+3] - 3.0*coeff[i+2] + 3.0*coeff[i+1] - coeff[i])/6.0 - \\
                       (e[i+3] + 27.0*e[i+2] + 93.0*e[i+1] + 59.0*e[i])/30.0
                c[i] = 0.5*(coeff[i+2] + coeff[i]) - coeff[i+1] - 0.5*d[i+1] - 2.5*d[i] - \\
                       0.1*(e[i+2] + 18.0*e[i+1] + 31.0*e[i])
                b[i] = coeff[i+1] - coeff[i] - c[i] - d[i] - 0.2*(4.0*e[i] + e[i+1])
            
            # FORTRAN: Boundary handling (lines 123-129)
            for i in range(n-3, n):
                b[i] = b[i-1] + 2.0*c[i-1] + 3.0*d[i-1] + 4.0*e[i-1] + 5.0*f[i-1]
                c[i] = c[i-1] + 3.0*d[i-1] + 6.0*e[i-1] + 10.0*f[i-1]
                d[i] = d[i-1] + 4.0*e[i-1] + 10.0*f[i-1]
                if i < n-1:
                    f[i] = coeff[i+1] - coeff[i] - b[i] - c[i] - d[i] - e[i]
            f[n-1] = f[n-2]
            
            # FORTRAN: Scaling (lines 131-140)
            fac = 1.0 / h_step
            for i in range(n):
                coeff[n + i] = b[i] * fac
            fac = fac / h_step
            for i in range(n):
                coeff[2*n + i] = c[i] * fac
            fac = fac / h_step
            for i in range(n):
                coeff[3*n + i] = d[i] * fac
            fac = fac / h_step
            for i in range(n):
                coeff[4*n + i] = e[i] * fac
            fac = fac / h_step
            for i in range(n):
                coeff[5*n + i] = f[i] * fac
    '''
    
    return quintic_code.strip()

if __name__ == "__main__":
    code = generate_exact_quintic_implementation()
    print("Generated exact quintic implementation:")
    print(code)