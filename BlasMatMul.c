#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"

void matmul_cblas_float(const float *A, const float *B, float *C, size_t n) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, A, n, B, n, 0, C, n);
}
