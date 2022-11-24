#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <string.h>


void matmul_plain_float(const float *A, const float *B, float *C, size_t n, size_t m, size_t p) {
    memset(C, 0, sizeof(float) * n * p);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++)
            for (size_t k = 0; k < p; k++)
                C[i * n + j] += A[i * n + k] * B[k * m + j];
}

void matmul_plain_ikj_float(const float *A, const float *B, float *C, size_t n, size_t m, size_t p) {
    memset(C, 0, sizeof(float) * n * p);
    for (size_t i = 0; i < n; i++)
        for (size_t k = 0; k < p; k++)
            for (size_t j = 0; j < m; j++)
                C[i * n + j] += A[i * n + k] * B[k * m + j];
}

void matmul_omp_float(const float *A, const float *B, float *C, size_t n, size_t m, size_t p) {
    memset(C, 0, sizeof(float) * n * p);
    omp_set_num_threads(omp_get_num_procs());
#pragma omp parallel for schedule(dynamic) default(none) shared(A, B, C, n, m, p)
    for (size_t i = 0; i < n; i++)
        for (size_t k = 0; k < p; k++)
            for (size_t j = 0; j < m; j++)
                C[i * n + j] += A[i * n + k] * B[k * m + j];
}

void matmul_omp_unroll_square_float(const float *A, const float *B, float *C, size_t n) {
    memset(C, 0, sizeof(float) * n * n);
    omp_set_num_threads(omp_get_num_procs());
#pragma omp parallel for schedule(dynamic) default(none) shared(A, B, C, n)
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k+=8) {
            float a1 = A[i * n + k];
            float a2 = A[i * n + k + 1];
            float a3 = A[i * n + k + 2];
            float a4 = A[i * n + k + 3];
            float a5 = A[i * n + k + 4];
            float a6 = A[i * n + k + 5];
            float a7 = A[i * n + k + 6];
            float a8 = A[i * n + k + 7];
            for (size_t j = 0; j < n; j+=8) {
                C[i * n + j] += a1 * B[k * n + j];
                C[i * n + j + 1] += a2 * B[(k + 1) * n + j + 1];
                C[i * n + j + 2] += a3 * B[(k + 2) * n + j + 2];
                C[i * n + j + 3] += a4 * B[(k + 3) * n + j + 3];
                C[i * n + j + 4] += a5 * B[(k + 4) * n + j + 4];
                C[i * n + j + 5] += a6 * B[(k + 5) * n + j + 5];
                C[i * n + j + 6] += a7 * B[(k + 6) * n + j + 6];
                C[i * n + j + 7] += a8 * B[(k + 7) * n + j + 7];
            }
        }
    }
}

extern inline size_t hsum_epi32(__m256i x){
    __m128i l = _mm256_extracti128_si256(x, 0);
    __m128i h = _mm256_extracti128_si256(x, 1);
    l         = _mm_add_epi32(l, h);
    l         = _mm_hadd_epi32(l, l);
    return _mm_extract_epi32(l, 0) + _mm_extract_epi32(l, 1);
}
extern inline float hsum128_ps(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums);
    sums        = _mm_add_ss(sums, shuf);
    return        _mm_cvtss_f32(sums);
}
extern inline float hsum256_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow  = _mm_add_ps(vlow, vhigh);
    return hsum128_ps(vlow);
}

void matmul_avx2_omp_square_float(const float *A, const float *B, float *C, size_t n) {
    memset(C, 0, sizeof(float) * n * n);
    __m256 sum = _mm256_setzero_ps();
    __m256 vec1_B = _mm256_setzero_ps();
    __m256 vec2_B = _mm256_setzero_ps();
    __m256 vec3_B = _mm256_setzero_ps();
    __m256 vec4_B = _mm256_setzero_ps();
    __m256 vecC = _mm256_setzero_ps();
    __m256 scalar1 = _mm256_setzero_ps();
    __m256 scalar2 = _mm256_setzero_ps();
    __m256 scalar3 = _mm256_setzero_ps();
    __m256 scalar4 = _mm256_setzero_ps();
    omp_set_num_threads(omp_get_num_procs());
#pragma omp parallel for schedule(dynamic) default(none) private(vec1_B,vec2_B,vec3_B,vec4_B,vecC,sum,scalar1,scalar2,scalar3,scalar4) shared(A,B,C,n)
    for (size_t i = 0; i < n; i++)
    {
        for (size_t k = 0; k < n; k+=4)
        {
            scalar1 = _mm256_set1_ps(A[i * n + k]);
            scalar2 = _mm256_set1_ps(A[i * n + k + 1]);
            scalar3 = _mm256_set1_ps(A[i * n + k + 2]);
            scalar4 = _mm256_set1_ps(A[i * n + k + 3]);
            for (size_t j = 0; j < n; j+=8)
            {
                vec1_B = _mm256_load_ps((B + (k * n + j)));
                vec1_B = _mm256_mul_ps(vec1_B, scalar1);
                vec2_B = _mm256_load_ps((B + ((k + 1) * n + j)));
                vec2_B = _mm256_mul_ps(vec2_B, scalar2);
                vec3_B = _mm256_load_ps((B + ((k + 2) * n + j)));
                vec3_B = _mm256_mul_ps(vec3_B, scalar3);
                vec4_B = _mm256_load_ps((B + ((k + 3) * n + j)));
                vec4_B = _mm256_mul_ps(vec4_B, scalar4);
                vecC = _mm256_load_ps((C + (i * n + j)));
                vecC = _mm256_add_ps(vec1_B, vecC);
                vecC = _mm256_add_ps(vec2_B, vecC);
                vecC = _mm256_add_ps(vec3_B, vecC);
                vecC = _mm256_add_ps(vec4_B, vecC);
                _mm256_store_ps((C + (i * n + j)), vecC);
            }
        }
    }
}

void matmul_avx2_omp_square_ufloat(const float *A, const float *B, float *C, size_t n) {
    __m256 sum = _mm256_setzero_ps();
    __m256 vec1_B = _mm256_setzero_ps();
    __m256 vec2_B = _mm256_setzero_ps();
    __m256 vec3_B = _mm256_setzero_ps();
    __m256 vec4_B = _mm256_setzero_ps();
    __m256 vecC = _mm256_setzero_ps();
    __m256 scalar1 = _mm256_setzero_ps();
    __m256 scalar2 = _mm256_setzero_ps();
    __m256 scalar3 = _mm256_setzero_ps();
    __m256 scalar4 = _mm256_setzero_ps();
    omp_set_num_threads(omp_get_num_procs());
#pragma omp parallel for schedule(dynamic) default(none) private(vec1_B,vec2_B,vec3_B,vec4_B,vecC,sum,scalar1,scalar2,scalar3,scalar4) shared(A,B,C,n)
    for (size_t i = 0; i < n; i++)
    {
        for (size_t k = 0; k < n; k+=4)
        {
            scalar1 = _mm256_set1_ps(A[i * n + k]);
            scalar2 = _mm256_set1_ps(A[i * n + k + 1]);
            scalar3 = _mm256_set1_ps(A[i * n + k + 2]);
            scalar4 = _mm256_set1_ps(A[i * n + k + 3]);
            for (size_t j = 0; j < n; j+=8)
            {
                vec1_B = _mm256_loadu_ps((B + (k * n + j)));
                vec1_B = _mm256_mul_ps(vec1_B, scalar1);
                vec2_B = _mm256_loadu_ps((B + ((k + 1) * n + j)));
                vec2_B = _mm256_mul_ps(vec2_B, scalar2);
                vec3_B = _mm256_loadu_ps((B + ((k + 2) * n + j)));
                vec3_B = _mm256_mul_ps(vec3_B, scalar3);
                vec4_B = _mm256_loadu_ps((B + ((k + 3) * n + j)));
                vec4_B = _mm256_mul_ps(vec4_B, scalar4);
                vecC = _mm256_loadu_ps((C + (i * n + j)));
                vecC = _mm256_add_ps(vec1_B, vecC);
                vecC = _mm256_add_ps(vec2_B, vecC);
                vecC = _mm256_add_ps(vec3_B, vecC);
                vecC = _mm256_add_ps(vec4_B, vecC);
                _mm256_storeu_ps((C + (i * n + j)), vecC);
            }
        }
    }
}

void mulBlock(size_t blockLength, const float * b1, const float * b2, float * res) {
    omp_set_num_threads(omp_get_num_procs());
#pragma omp parallel for schedule(dynamic) default(none) shared(b1,b2,res,blockLength)
    for (size_t i=0; i<blockLength; i++) {
        for (size_t j=0; j<blockLength; j+=8) {
            for (size_t k=0; k<blockLength; k+=8) {
                __m256 vecA = _mm256_loadu_ps((b1 + (i * blockLength + k)));
                __m256 vecB1 = _mm256_loadu_ps((b2 + (j * blockLength + k)));
                __m256 vecB2 = _mm256_loadu_ps((b2 + ((j + 1) * blockLength + k)));
                __m256 vecB3 = _mm256_loadu_ps((b2 + ((j + 2) * blockLength + k)));
                __m256 vecB4 = _mm256_loadu_ps((b2 + ((j + 3) * blockLength + k)));
                __m256 vecB5 = _mm256_loadu_ps((b2 + ((j + 4) * blockLength + k)));
                __m256 vecB6 = _mm256_loadu_ps((b2 + ((j + 5) * blockLength + k)));
                __m256 vecB7 = _mm256_loadu_ps((b2 + ((j + 6) * blockLength + k)));
                __m256 vecB8 = _mm256_loadu_ps((b2 + ((j + 7) * blockLength + k)));
                vecB1 = _mm256_mul_ps(vecA, vecB1);
                vecB2 = _mm256_mul_ps(vecA, vecB2);
                vecB3 = _mm256_mul_ps(vecA, vecB3);
                vecB4 = _mm256_mul_ps(vecA, vecB4);
                vecB5 = _mm256_mul_ps(vecA, vecB5);
                vecB6 = _mm256_mul_ps(vecA, vecB6);
                vecB7 = _mm256_mul_ps(vecA, vecB7);
                vecB8 = _mm256_mul_ps(vecA, vecB8);
                __m256 sum = _mm256_setr_ps(hsum256_ps(vecB1), hsum256_ps(vecB2), hsum256_ps(vecB3), hsum256_ps(vecB4), hsum256_ps(vecB5), hsum256_ps(vecB6), hsum256_ps(vecB7), hsum256_ps(vecB8));
                __m256 vecC = _mm256_loadu_ps((res + i * blockLength + j ));
                vecC = _mm256_add_ps(vecC, sum);
                _mm256_storeu_ps((res + (i * blockLength + j)), vecC);
            }
        }
    }
}

void addBlock(size_t blockLength, const float *resBlock, float *C, size_t row, size_t col, size_t n) {
    omp_set_num_threads(omp_get_num_procs());
#pragma omp parallel for schedule(dynamic) default(none) shared(resBlock,C,blockLength,row,col,n)
    for (size_t i=0; i<blockLength; i++) {
        for (size_t j=0; j<blockLength; j+=8) {
            __m256 vecC = _mm256_loadu_ps((C + (row + i) * n + col + j));
            __m256 vecRes = _mm256_loadu_ps((resBlock + i * blockLength + j));
            vecC = _mm256_add_ps(vecC, vecRes);
            _mm256_storeu_ps((C + (row + i) * n + col + j), vecC);
        }
    }
}

void matmul_avx2_div_square_float(const float *A, const float *B, float *C, size_t n) {
    size_t MAXSIZE = 512;
    memset(C, 0, n * n * sizeof(float));
    size_t blockLength = n;
    if (n > MAXSIZE)
        blockLength = MAXSIZE;
    size_t blockSize = blockLength * blockLength;
    size_t blockNumber = n / blockLength; // number of blocks in one row or column

    float * ARearrange = (float *)malloc(n * n * sizeof(float));
    float * BRearrange = (float *)malloc(n * n * sizeof(float));
    size_t firstIndexK = 0;
    for (size_t k = 0; k<n; k+=blockLength) {
        for (size_t i = 0; i<n; i++) {
            for (size_t j = 0; j<blockLength;j++) {
                ARearrange[firstIndexK + i * blockLength + j] = A[i * n + k + j];
//                BRearrange[firstIndexK + i * blockLength + j] = BTrans[i * n + k + j];
                BRearrange[firstIndexK + i * blockLength + j] = B[(k + j) * n + i];
            }
        }
        firstIndexK += blockLength * n;
    }

#ifndef UNIX
    float * b1 = (float *)malloc(blockSize * sizeof(float));
    float * b2 = (float *)malloc(blockSize * sizeof(float));
    float * resBlock = (float *)malloc(blockSize * sizeof(float));
    float * blockGroupA = (float *)malloc(blockNumber * blockSize * sizeof(float));
    float * blockGroupB = (float *)malloc(blockNumber * blockSize * sizeof(float));
#else
    float * b1 = (float *)aligned_alloc(256, blockSize * sizeof(float));
    float * b2 = (float *)aligned_alloc(256, blockSize * sizeof(float));
    float * resBlock = (float *)aligned_alloc(256, blockSize * sizeof(float));
    float * blockGroupA = (float *)aligned_alloc(256, blockNumber * blockSize * sizeof(float));
    float * blockGroupB = (float *)aligned_alloc(256, blockNumber * blockSize * sizeof(float));
#endif


    for (size_t gr = 0; gr < blockNumber; gr++) { //group
        memcpy(blockGroupA, ARearrange + gr * blockSize * blockNumber, blockNumber * blockSize * sizeof(float));
        memcpy(blockGroupB, BRearrange + gr * blockSize * blockNumber, blockNumber * blockSize * sizeof(float));
        for (size_t ag = 0; ag < blockNumber; ag++) {
            memcpy(b1, blockGroupA + ag * blockSize, blockSize * sizeof(float));
            for (size_t bg = 0; bg < blockNumber; bg++) {
                memcpy(b2, blockGroupB + bg * blockSize, blockSize * sizeof(float));
                memset(resBlock, 0, blockSize * sizeof(float));
                mulBlock(blockLength, b1, b2, resBlock);
                addBlock(blockLength, resBlock, C, ag * blockLength, bg * blockLength, n);
            }
        }
    }
    free(ARearrange);
    free(BRearrange);
    free(b1);
    free(b2);
    free(resBlock);
    free(blockGroupA);
    free(blockGroupB);
}