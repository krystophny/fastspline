#ifndef BISPEV_WRAPPER_H
#define BISPEV_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * C wrapper for DIERCKX bispev Fortran routine
 * 
 * Evaluates a bivariate spline on a grid of points.
 * 
 * @param tx     Array of knots in x-direction (length nx)
 * @param nx     Number of knots in x-direction
 * @param ty     Array of knots in y-direction (length ny)
 * @param ny     Number of knots in y-direction
 * @param c      B-spline coefficients (length (nx-kx-1)*(ny-ky-1))
 * @param kx     Degree of spline in x-direction
 * @param ky     Degree of spline in y-direction
 * @param x      Array of x-coordinates to evaluate (length mx)
 * @param mx     Number of x-coordinates
 * @param y      Array of y-coordinates to evaluate (length my)
 * @param my     Number of y-coordinates
 * @param z      Output array for results (length mx*my)
 * @param wrk    Working space array (length >= mx*(kx+1)+my*(ky+1))
 * @param lwrk   Length of working space array
 * @param iwrk   Integer working space (length >= mx+my)
 * @param kwrk   Length of integer working space
 * @param ier    Error flag (0 = success, 10 = invalid input)
 * 
 * @return 0 on success, non-zero on error
 */
int bispev_c(const double* tx, int nx, const double* ty, int ny,
             const double* c, int kx, int ky,
             const double* x, int mx, const double* y, int my,
             double* z, double* wrk, int lwrk, int* iwrk, int kwrk,
             int* ier);

/* Fortran subroutine declaration (compiled with -fno-underscoring) */
extern void bispev(const double* tx, const int* nx, const double* ty, const int* ny,
                   const double* c, const int* kx, const int* ky,
                   const double* x, const int* mx, const double* y, const int* my,
                   double* z, double* wrk, const int* lwrk, int* iwrk, const int* kwrk,
                   int* ier);

#ifdef __cplusplus
}
#endif

#endif /* BISPEV_WRAPPER_H */