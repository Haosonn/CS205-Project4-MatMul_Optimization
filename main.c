#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <omp.h>
#include <string.h>
#include "BlasMatMul.h"
#include "Matmul.h"

void endTime(const char * funcName, size_t size, double start, float * C) {
    double end = omp_get_wtime();
    (void) !freopen("../log.txt", "a", stdout);
    printf("%s", funcName);
    printf(" takes %fs\n", (double) (end - start));
    char addr[100];
    sprintf(addr, "../%s.txt", funcName);
    (void) !freopen(addr, (size == 16)?"w":"a", stdout);
    for (size_t i = 0; i < size; i++) printf("%f\n", C[i * size]);
    (void) !freopen("../log.txt", "a", stdout);
}

void floatTest();


size_t size[] = {16,128,1024,2048, 4096,8192};

int main() {
    srand(1);
    (void) !freopen("../log.txt", "w", stdout);
    floatTest();
}

void floatTest(){
    double start, end;
    float *Amem, *Bmem, *Cmem, *A, *B, *C;
    for (size_t i = 0; i <= 5; i++) {
        printf("Matrix size: %ld\n", size[i]);
        Amem = (float *) malloc((size[i] * size[i] + 255) * sizeof(float));
        Bmem = (float *) malloc((size[i] * size[i] + 255) * sizeof(float));
        Cmem = (float *) malloc((size[i] * size[i] + 255) * sizeof(float));
        A = (float *) (((uintptr_t) Amem + 255) & ~(uintptr_t) 0xFF);
        B = (float *) (((uintptr_t) Bmem + 255) & ~(uintptr_t) 0xFF);
        C = (float *) (((uintptr_t) Cmem + 255) & ~(uintptr_t) 0xFF);
//        A = Amem;
//        B = Bmem;
//        C = Cmem;
        size_t size_ = size[i] * size[i];
        for (size_t j = 0; j < size_; j++) {
            A[j] = (float) (rand() % 100) / (rand() % 5 + 1);
            B[j] = (float) (rand() % 100) / (rand() % 5 + 1);
            C[j] = 0;
        }

#ifdef UNIX
        start = omp_get_wtime();
        matmul_cblas_float(A, B, C, size[i]);
        endTime("cblas", size[i], start, C);
#endif

//        start = omp_get_wtime();
//        matmul_omp_float(A, B, C, size[i], size[i], size[i]);
//        endTime("omp", size[i], start, C);

        start = omp_get_wtime();
        matmul_avx2_div_square_float(A, B, C, size[i]);
        endTime("avx2_div_square", size[i], start, C);

//        start = omp_get_wtime();
//        matmul_avx2_omp_square_float(A, B, C, size[i]);
//        endTime("avx2_omp_square", size[i], start, C);

        free(Amem);
        free(Bmem);
        free(Cmem);
    }
}
