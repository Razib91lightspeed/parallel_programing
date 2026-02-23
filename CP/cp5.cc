#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(1);                                                \
    }                                                           \
} while (0)

#define TILE 32


#define NORM_BLOCK 256

__global__ void normalize_mean(int ny, int nx,
                               const float *__restrict__ data,
                               float *__restrict__ X)
{
    int y = blockIdx.x;
    if (y >= ny) return;

    __shared__ float buf[NORM_BLOCK];

    /* Partial sum over columns */
    float sum = 0.0f;
    for (int x = threadIdx.x; x < nx; x += blockDim.x)
        sum += data[x + y * nx];

    buf[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            buf[threadIdx.x] += buf[threadIdx.x + stride];
        __syncthreads();
    }

    float mean = buf[0] / nx;


    for (int x = threadIdx.x; x < nx; x += blockDim.x)
        X[x + y * nx] = data[x + y * nx] - mean;
}

__global__ void normalize_norm(int ny, int nx,
                               float *__restrict__ X)
{
    int y = blockIdx.x;
    if (y >= ny) return;

    __shared__ float buf[NORM_BLOCK];
    float sum = 0.0f;
    for (int x = threadIdx.x; x < nx; x += blockDim.x) {
        float v = X[x + y * nx];
        sum += v * v;
    }

    buf[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            buf[threadIdx.x] += buf[threadIdx.x + stride];
        __syncthreads();
    }

    float invnorm = rsqrtf(buf[0]);

    for (int x = threadIdx.x; x < nx; x += blockDim.x)
        X[x + y * nx] *= invnorm;
}

__global__ void correlate_tiled(int ny, int nx,
                                const float *__restrict__ X,
                                float *__restrict__ result)
{
    if (blockIdx.y < blockIdx.x) return;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int i = blockIdx.y * TILE + threadIdx.y;
    int j = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < nx; k += TILE) {

        if (i < ny && k + threadIdx.x < nx)
            As[threadIdx.y][threadIdx.x] =
                __ldg(&X[(k + threadIdx.x) + i * nx]);
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (j < ny && k + threadIdx.y < nx)
            Bs[threadIdx.y][threadIdx.x] =
                __ldg(&X[(k + threadIdx.y) + j * nx]);
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int t = 0; t < TILE; t++)
            sum += As[threadIdx.y][t] * Bs[t][threadIdx.x];

        __syncthreads();
    }

    if (i < ny && j <= i)
        result[i + j * ny] = sum;
}

void correlate(int ny, int nx, const float *data, float *result)
{
    float *d_data = nullptr;
    float *d_X = nullptr;
    float *d_result = nullptr;

    size_t data_size   = (size_t)ny * nx * sizeof(float);
    size_t result_size = (size_t)ny * ny * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_data, data_size));
    CUDA_CHECK(cudaMalloc(&d_X, data_size));
    CUDA_CHECK(cudaMalloc(&d_result, result_size));
    CUDA_CHECK(cudaMemset(d_result, 0, result_size));

    CUDA_CHECK(cudaMemcpy(d_data, data, data_size,
                          cudaMemcpyHostToDevice));

    dim3 block1(NORM_BLOCK);
    dim3 grid1(ny);

    normalize_mean<<<grid1, block1>>>(ny, nx, d_data, d_X);
    CUDA_CHECK(cudaGetLastError());

    normalize_norm<<<grid1, block1>>>(ny, nx, d_X);
    CUDA_CHECK(cudaGetLastError());
    dim3 block2(TILE, TILE);
    dim3 grid2((ny + TILE - 1) / TILE,
               (ny + TILE - 1) / TILE);

    correlate_tiled<<<grid2, block2>>>(ny, nx, d_X, d_result);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(result, d_result, result_size,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_result));
}
