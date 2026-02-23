#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

__global__ void normalize_mean(int ny, int nx,
                               const float *data,
                               float *X)
{
    int y = blockIdx.x;
    if (y >= ny) return;

    float sum = 0.0f;

    for (int x = threadIdx.x; x < nx; x += blockDim.x)
        sum += data[x + y * nx];

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    __shared__ float mean;
    if (threadIdx.x == 0)
        mean = sum / nx;

    __syncthreads();

    for (int x = threadIdx.x; x < nx; x += blockDim.x)
        X[x + y * nx] = data[x + y * nx] - mean;
}

__global__ void normalize_norm(int ny, int nx, float *X)
{
    int y = blockIdx.x;
    if (y >= ny) return;

    float sum = 0.0f;

    for (int x = threadIdx.x; x < nx; x += blockDim.x) {
        float v = X[x + y * nx];
        sum += v * v;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    __shared__ float invnorm;
    if (threadIdx.x == 0)
        invnorm = rsqrtf(sum);

    __syncthreads();

    for (int x = threadIdx.x; x < nx; x += blockDim.x)
        X[x + y * nx] *= invnorm;
}

__global__ void correlate_kernel(int ny, int nx,
                                 const float *X,
                                 float *result)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= ny || j > i) return;

    float sum = 0.0f;
    for (int x = 0; x < nx; x++)
        sum += X[x + i * nx] * X[x + j * nx];

    result[i + j * ny] = sum;
}


void correlate(int ny, int nx, const float *data, float *result) {
    float *d_data = NULL;
    float *d_X = NULL;
    float *d_result = NULL;

    size_t data_size   = (size_t)ny * nx * sizeof(float);
    size_t result_size = (size_t)ny * ny * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_data, data_size));
    CUDA_CHECK(cudaMalloc(&d_X, data_size));
    CUDA_CHECK(cudaMalloc(&d_result, result_size));

    CUDA_CHECK(cudaMemset(d_result, 0, result_size));

    CUDA_CHECK(cudaMemcpy(d_data, data, data_size,
                          cudaMemcpyHostToDevice));
    dim3 block1(32);
    dim3 grid1(ny);

    normalize_mean<<<grid1, block1>>>(ny, nx, d_data, d_X);
    CUDA_CHECK(cudaGetLastError());

    normalize_norm<<<grid1, block1>>>(ny, nx, d_X);
    CUDA_CHECK(cudaGetLastError());
    dim3 block2(16, 16);
    dim3 grid2((ny + 15) / 16, (ny + 15) / 16);

    correlate_kernel<<<grid2, block2>>>(ny, nx, d_X, d_result);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(result, d_result, result_size,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_result));
}
