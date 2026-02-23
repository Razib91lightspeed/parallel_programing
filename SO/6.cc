#include <cuda_runtime.h>
#include <cstdlib>
#include <algorithm>

typedef unsigned long long data_t;

#define BLOCK_SIZE 256


__global__ void predicate_kernel(const data_t* input,
                                 unsigned int* predicate,
                                 int bit,
                                 int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        predicate[i] = (input[i] >> bit) & 1ull;
    }
}

__global__ void scan_kernel(const unsigned int* in,
                            unsigned int* out,
                            unsigned int* block_sums,
                            int N)
{
    __shared__ unsigned int temp[BLOCK_SIZE * 2];

    int tid = threadIdx.x;
    int start = 2 * blockIdx.x * blockDim.x;


    temp[tid] = (start + tid < N) ? in[start + tid] : 0;
    temp[tid + blockDim.x] =
        (start + tid + blockDim.x < N) ? in[start + tid + blockDim.x] : 0;

    for (int stride = 1; stride < 2 * blockDim.x; stride <<= 1) {
        __syncthreads();
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < 2 * blockDim.x)
            temp[idx] += temp[idx - stride];
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = temp[2 * blockDim.x - 1];
        temp[2 * blockDim.x - 1] = 0;
    }
    for (int stride = blockDim.x; stride > 0; stride >>= 1) {
        __syncthreads();
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < 2 * blockDim.x) {
            unsigned int t = temp[idx - stride];
            temp[idx - stride] = temp[idx];
            temp[idx] += t;
        }
    }
    __syncthreads();
    if (start + tid < N)
        out[start + tid] = temp[tid];
    if (start + tid + blockDim.x < N)
        out[start + tid + blockDim.x] = temp[tid + blockDim.x];
}
__global__ void add_block_offsets(unsigned int* data,
                                  const unsigned int* block_offsets,
                                  int N)
{
    int tid = threadIdx.x;
    int block = blockIdx.x;

    unsigned int offset = block_offsets[block];

    int i1 = 2 * block * blockDim.x + tid;
    int i2 = i1 + blockDim.x;

    if (i1 < N) data[i1] += offset;
    if (i2 < N) data[i2] += offset;
}
__global__ void scatter_kernel(const data_t* input,
                               data_t* output,
                               const unsigned int* predicate,
                               const unsigned int* scan,
                               int total_zeros,
                               int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    unsigned int bit = predicate[i];
    unsigned int pos = (bit == 0)
        ? (i - scan[i])
        : (total_zeros + scan[i]);

    output[pos] = input[i];
}
void psort(int n, data_t* data)
{
    if (n <= 1) return;
    data_t *d_in, *d_out;
    unsigned int *predicate, *scan, *block_sums;

    cudaMalloc(&d_in, n * sizeof(data_t));
    cudaMalloc(&d_out, n * sizeof(data_t));
    cudaMalloc(&predicate, n * sizeof(unsigned int));
    cudaMalloc(&scan, n * sizeof(unsigned int));
    int num_blocks = (n + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);
    cudaMalloc(&block_sums, num_blocks * sizeof(unsigned int));
    cudaMemcpy(d_in, data, n * sizeof(data_t),
               cudaMemcpyHostToDevice);
    unsigned int* h_block_sums =
        (unsigned int*)malloc(num_blocks * sizeof(unsigned int));
    for (int bit = 0; bit < 64; bit++) {
        predicate_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                           BLOCK_SIZE>>>(
            d_in, predicate, bit, n);
        scan_kernel<<<num_blocks, BLOCK_SIZE>>>(
            predicate, scan, block_sums, n);
        cudaMemcpy(h_block_sums, block_sums,
                   num_blocks * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);

        unsigned int running = 0;
        for (int i = 0; i < num_blocks; i++) {
            unsigned int tmp = h_block_sums[i];
            h_block_sums[i] = running;
            running += tmp;
        }

        cudaMemcpy(block_sums, h_block_sums,
                   num_blocks * sizeof(unsigned int),
                   cudaMemcpyHostToDevice);
        add_block_offsets<<<num_blocks, BLOCK_SIZE>>>(
            scan, block_sums, n);
        unsigned int last_pred, last_scan;
        cudaMemcpy(&last_pred, predicate + n - 1,
                   sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_scan, scan + n - 1,
                   sizeof(unsigned int), cudaMemcpyDeviceToHost);

        int total_zeros = n - (last_pred + last_scan);
        scatter_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                         BLOCK_SIZE>>>(
            d_in, d_out, predicate, scan, total_zeros, n);
        std::swap(d_in, d_out);
    }
    cudaMemcpy(data, d_in, n * sizeof(data_t),
               cudaMemcpyDeviceToHost);

    free(h_block_sums);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(predicate);
    cudaFree(scan);
    cudaFree(block_sums);
}
