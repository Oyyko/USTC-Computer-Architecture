# 体系结构实验5

PB19000304 张佳

[TOC]

## 实验简介 

---

数据级并⾏是在计算机体系结构课程中重点讨论的⼀种并⾏⽅式，本实验将在CPU与GPU平台上开展，以矩阵乘法这⼀ 经典的例⼦作为切⼊点，动⼿实现不同版本的矩阵乘法实现，探讨不同矩阵规模与划分参数的性能。 

## 实验约定 

为了避免复杂，本次试验涉及到的矩阵运算的规模 ，每个程序的矩阵规模需要由参数传⼊程序，以便考察不同规模的矩阵乘法的性能。 

为了消除编译器优化的影响，在CPU与GPU平台上的编译优化参数可以⾃⾏选择，你需要提供 Makefile 来辅助 编译你的程序。 

在CPU与GPU平台上的实验的数据类型为 float32 。 在CPU平台上计时请包含 time.h ⽂件，使⽤ clock() 函数计时；

在GPU上请使⽤ nvprof ⼯具对你写的矩阵乘法kernel的时间进⾏profiling。 

在CPU平台上请使⽤动态内存分配申请矩阵的空间（为⼀维数组形式），随机数初始化两个参与计算的矩阵 和 ，随机初始化的⽬的是为了验证计算结果的正确性；

在GPU上请先在Host端使⽤动态内存分配申请矩阵的空间 （为⼀维数组形式），随机数初始化两个参与计算的矩阵和 ，随机初始化的⽬的是为了验证计算结果的正确性。 

本实验⽆线下检查环节，请各位同学将实验源代码与实验报告打包上传。

## CPU 任务1 基础矩阵乘法

```c
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
    float *C = malloc(sizeof(float) * N * N);
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
```

如图 这里还没有设置随机数种子，一会儿设置。

采用同样的种子，即可用这个算法验证别的算法的正确性。

（任务2中并不是和任务1的结果对比，而是直接用两个函数比较的，所以没必要了）

我们加上生成随机种子的：

```c
srand((unsigned)time(NULL));
```

即可

还可以优化如下：

改变一下运算的顺序，使得调用内存变得连续，从而加快运算速度

```c
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
```

![image-20220526161622233](/home/zhangjia/.config/Typora/typora-user-images/image-20220526161622233.png)

可以看到确实加快了运行。

## CPU 任务2 AVX矩阵乘法

```c
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
                _mm256_broadcast_ss(A + i * N + k);
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
```

 上面是实现的AVX矩阵乘法

AVX是一种用SIMD方式支持并行计算的指令集。Intel CPU会实现这个指令集。这里我们用C语言调用相关的指令避免直接编写汇编。相关定义在`#include <immintrin.h>`中。

```c
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
```

我们首先定义一个`tmp`变量用来存放mm256格式的浮点数，一个float是32bit 所以256个bit是8个float。 tmp里面相当于存放了8个float。

我们这里对j进行合并，按照j的顺序把8个float合并为一个mm256，之后再进行矩阵乘法

```c
tmp = _mm256_add_ps(tmp, _mm256_mul_ps(_mm256_broadcast_ss(A + i * N + k), _mm256_loadu_ps(B + k * N + j)));
                _mm256_storeu_ps(C + i * N + j, tmp);
```

这两行。第一行计算相乘的结果并加到tmp上面，由于合并之后的8个`k，j`，在这里都乘以同一个`i,k`对应的值，因此用`_mm256_broadcast_ss`。 它的作用是把一个float复制8份放入mm256中去。这样我们便实现了一个指令计算8个数字相乘，SIMD。 从而加快了运行速度。

### 验证正确性

这里我开了C1，C2两个结果数组。分别用AVX和普通矩阵乘法进行计算，然后比较它们是否相同，如果不同就报错并结束程序。

```c
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
```

结果如下：

![image-20220527165721582](/home/zhangjia/.config/Typora/typora-user-images/image-20220527165721582.png)

## CPU 任务 3 AVX分块矩阵乘法

先前的AVX实现由于仍然是三重循环为主体，访存跨度较⼤，并未充分利⽤cache的局部性。 

你需要使⽤CPU-任务1中的基础矩阵乘法验证计算的正确性，验证结束后在性能测量阶段可以不进⾏正确性的验证。 

在本任务中，你需要调研基于AVX指令集的分块矩阵乘法的实现，并完成代码，你可能需要对B矩阵进⾏转置。



分块优化的目标是通过对输入数据进行分块尽可能使内存访问连续，提高cache命中率，进而提升程序的整体性能。

实现如下：

```c
#define BLOCKSIZE 32

void gemm_avx_block_32(int ii, int jj, int kk, float *A, float *B, float *C)
{
    __m256 tmp;
    for (int i = ii; i < ii + BLOCKSIZE; ++i)
    {
        for (int j = jj; j < jj + BLOCKSIZE; j += 8)
        {
            tmp = _mm256_loadu_ps(C + i * N + j); //取出16个float
            for (int k = kk; k < kk + BLOCKSIZE; ++k)
            {
                tmp = _mm256_add_ps(tmp, _mm256_mul_ps(_mm256_broadcast_ss(A + i * N + k), _mm256_loadu_ps(B + k * N + j)));
            }
            _mm256_storeu_ps(C + i * N + j, tmp);
        }
    }
}

void gemm_avx_block(float *A, float *B, float *C)
{
    for (int ii = 0; ii < N; ii += BLOCKSIZE) // 遍历所有行
    {
        for (int jj = 0; jj < N; jj += BLOCKSIZE) // 遍历所有列
        {
            for (int kk = 0; kk < N; kk += BLOCKSIZE) // 分块
            {
                gemm_avx_block_32(ii, jj, kk, A, B, C);
            }
        }
    }
}
```

程序正确性测试如下：

```c
start_time = clock();
    gemm_avx_block(A, B, C3);
    end_time = clock();
    printf("gemm avx blockTIME: %.20lf\n\n", (end_time - start_time) / (CLOCKS_PER_SEC + 0.0));

    gemm_verify(C1, C3);
```

`gemm_verify`在检测到 普通矩阵乘法算出的C1和AVX_BLOCK算出的C3不一样的时候会调用`exit(1)`

并且如果正确的话 会输出YES YOU'RE RIGHT!

```c
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
```



运行结果如下：

![image-20220528152548546](/home/zhangjia/.config/Typora/typora-user-images/image-20220528152548546.png)

可以看到结果正确 并且Block版本比普通的AVX优化版本要更快

## CPU 报告

### 对不同规模的输⼊（输⼊的范围由你⾃⼰确定，可以不考虑过⼤的矩阵规模，因为这可能导致性能测量很慢）， 考察三种实现的性能差异，并简要分析原因。

修改源代码可以手动输入矩阵规模大小N

```c
int main(int argc, char **argv)
{
    N = (1 << (argv[1][0] - '0'));
```

![image-20220528153847485](/home/zhangjia/.config/Typora/typora-user-images/image-20220528153847485.png)

数据如上图

可以看到，AVX版本和AVX BLOCK版本都比普通版本少一个数量级。

AVX BLOCK版本又比AVX版本快了大约1倍

这是因为：

AVX版本使用特殊的指令，在硬件层面加速，SIMD使得一个指令可以操作多个数据进行运算，从而实现了硬件层面的并行，从而加快了运行速度。

AVX BLOCK版本更好的利用了缓存，因此缓存命中率更高，因此更快。

### 对CPU-任务3中的AVX分块矩阵乘法，探讨不同的分块参数对性能的影响，并简要分析原因。

![image-20220528160452613](/home/zhangjia/.config/Typora/typora-user-images/image-20220528160452613.png)



![image-20220528160534236](/home/zhangjia/.config/Typora/typora-user-images/image-20220528160534236.png)

参数为BLOCKSIZE等于输入的参数x对应的$2^x$大小

x为1，2的时候BLOCKSIZE小于8，而`_mm256`对应8个float从而导致无法正确计算，则答案错误。

分块的目的在于一个块内的计算都可以被缓存，从而需要块的大小小于cache大小。

从上面可以看到，当块的大小小于cache大小的时候，随着BLOCKSIZE增大，时间变化不大。而当BLOCKSIZE增大到大于cache大小的时候，时间就会变大较多。

### 调研并了解CPU平台上其它矩阵乘法的优化⼿段，在报告中简要总结。

循环重排序（reordering）

通过调整循环顺序为i，k，j来使得空间访问局部性良好，从而加快

写缓存优化（write caching）

对C的每个块的计算结果，我们在写回的时候，也是跳跃式的逐段写回，理想的情况应该是一个块的计算结果在内存中是连续的放置的，所以我们需要再开一块write cache内存空间，然后每个block的计算结果直接在write cache上读写，最后计算完一个block之后，整块写回C数组中对应的不同数组段上。

循环展开（UNROLL）

直接进行循环展开来加速。

并行计算：

使用MPI实现多进程的并行计算，或者使用OPENMP，`pthread`库实现多线程的方式来进行并行计算，利用CPU的多个核，来加快运行。

## GPU 任务

实现代码如下：

```c
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
```



## GPU 报告

