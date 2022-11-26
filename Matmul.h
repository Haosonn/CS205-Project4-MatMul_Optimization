#pragma once

#ifndef MATRIXLIB_MATMUL_H
#define MATRIXLIB_MATMUL_H


void matmul_plain_float(const float *A, const float *B, float *C, size_t n, size_t m, size_t p);

void matmul_plain_ikj_float(const float *A, const float *B, float *C, size_t n, size_t m, size_t p);

void matmul_omp_float(const float *A, const float *B, float *C, size_t n, size_t m, size_t p);

void matmul_omp_unroll_square_float(const float *A, const float *B, float *C, size_t n);

void matmul_avx2_omp_square_float(const float *A, const float *B, float *C, size_t n);

void matmul_avx2_omp_square_ufloat(const float *A, const float *B, float *C, size_t n);

void matmul_transposed_omp_square_float(const float *A, const float *B, float *C, size_t n);

void matmul_avx2_div_square_float(const float *A, const float *B, float *C, size_t n);

void matmul_improved_sqaure_float(const float *A, const float *B, float *C, size_t n);

#endif
