#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
        for (int k = 0; k < N; ++k)
        {
            float tmp = A[i * N + k];
            for (int j = 0; j < N; ++j)
            {
                C[i * N + j] += tmp * B[k * N + j];
            }
        }
    }
}

int main()
{
    float *A = malloc(sizeof(float) * N * N);
    float *B = malloc(sizeof(float) * N * N);
    float *C = malloc(sizeof(float) * N * N);
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
        C[i] = 0;
    }
    clock_t start_time = clock();
    gemm_baseline(A, B, C);
    clock_t end_time = clock();
    printf("TIME: %.20lf\n\n", (end_time - start_time) / (CLOCKS_PER_SEC + 0.0));
    // print_array(A);
    // print_array(B);
    // print_array(C);
    free(A);
    free(B);
    free(C);
    return 0;
}
