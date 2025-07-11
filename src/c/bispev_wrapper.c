#include "bispev_wrapper.h"
#include <string.h>

int bispev_c(const double* tx, int nx, const double* ty, int ny,
             const double* c, int kx, int ky,
             const double* x, int mx, const double* y, int my,
             double* z, double* wrk, int lwrk, int* iwrk, int kwrk,
             int* ier) {
    
    /* Call the Fortran subroutine
     * Fortran expects all parameters to be passed by reference
     */
    bispev(tx, &nx, ty, &ny, c, &kx, &ky, x, &mx, y, &my,
           z, wrk, &lwrk, iwrk, &kwrk, ier);
    
    return *ier;
}