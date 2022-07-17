#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

int N = (1 << 8);

void print_array(float *X)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            printf("%lf ", X[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void gemm_baseline(float *A, float *B, float *C)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            for (int k = 0; k < N; ++k)
            {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void gemm_verify(float *C1, float *C2)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (C1[i * N + j] != C2[i * N + j])
            {
                printf("WRONG!\n");
                exit(1);
            }
        }
    }
    printf("YES! YOU'RE RIGHT!\n");
}

void gemm_avx(float *A, float *B, float *C)
{
    __m256 tmp;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; j += 8)
        {
            tmp = _mm256_loadu_ps(C + i * N + j);

            for (int k = 0; k < N; ++k)
            {
                // printf("I:%d,J:%d,K:%d\n", i, j, k);
                tmp = _mm256_add_ps(tmp, _mm256_mul_ps(_mm256_broadcast_ss(A + i * N + k), _mm256_loadu_ps(B + k * N + j)));
                _mm256_storeu_ps(C + i * N + j, tmp);
            }
        }
    }
}

int main()
{
    float *A = malloc(sizeof(float) * N * N);
    float *B = malloc(sizeof(float) * N * N);
    float *C1 = malloc(sizeof(float) * N * N);
    float *C2 = malloc(sizeof(float) * N * N);
    srand((unsigned)time(NULL));
    for (int i = 0; i < N * N; ++i)
    {
        A[i] = rand() / (double)(RAND_MAX);
    }
    for (int i = 0; i < N * N; ++i)
    {
        B[i] = rand() / (double)(RAND_MAX);
    }
    for (int i = 0; i < N * N; ++i)
    {
        C1[i] = 0;
        C2[i] = 0;
    }
    gemm_baseline(A, B, C1);
    clock_t start_time = clock();
    gemm_avx(A, B, C2);
    clock_t end_time = clock();
    gemm_verify(C1, C2);
    printf("TIME: %.20lf\n\n", (end_time - start_time) / (CLOCKS_PER_SEC + 0.0));
    // print_array(A);
    // print_array(B);
    // print_array(C);
    free(A);
    free(B);
    free(C1);
    free(C2);
    return 0;
}
