#include <stdio.h>
#include <stdlib.h>

#define N (1 << 10)

__global__ void gemm_baseline(float *A, float *B, float *C)
{
    int rr = blockIdx.x * blockDim.x + threadIdx.x;
    int cc = blockIdx.y * blockDim.y + threadIdx.y;

    if ((rr < N) && (cc < N))
    {
        float t = 0;
        for (int i = 0; i < N; i++)
        {
            t += A[rr * N + i] * B[i * N + cc];
        }
        C[rr * N + cc] = t;
    }
    return;
}

#define BLOCK_SIZE 32

__global__ void gemm_cuda_block(float *A, float *B, float *C)
{
    __shared__ float Ads[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bds[BLOCK_SIZE][BLOCK_SIZE];
    int rr = by * BLOCK_SIZE + ty, cc = bx * BLOCK_SIZE + tx, tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y;
    float sum = 0;
    for (int i = 0; i < N / BLOCK_SIZE; i++)
    {
        Ads[ty][tx] = A[rr * N + i * BLOCK_SIZE + tx];
        Bds[ty][tx] = B[cc + N * (ty + BLOCK_SIZE * i)];
        __syncthreads();
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
            sum += Ads[ty][j] * Bds[j][tx];
            __syncthreads();
        }
    }

    C[rr * N + cc] = sum;
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

void gemm_normal(float *A, float *B, float *C)
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

int main()
{
    float *A = malloc(sizeof(float) * N * N);
    float *B = malloc(sizeof(float) * N * N);
    float *C1 = malloc(sizeof(float) * N * N);
    float *C2 = malloc(sizeof(float) * N * N);
    float *C3 = malloc(sizeof(float) * N * N);
    // srand((unsigned)time(NULL));
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
        C3[i] = 0;
    }
    clock_t start_time, end_time;
    gemm_normal(A, B, C1);
    gemm_baseline(A, B, C2);
    gemm_cuda_block(A, B, C2);

    gemm_verify(C1, C2);
    free(A);
    free(B);
    free(C1);
    free(C2);
    free(C3);
    return 0;
}