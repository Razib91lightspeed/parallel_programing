#include <vector>
#include <cfloat>
#include <cuda_runtime.h>

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

// ---------------- CUDA kernel ----------------
__global__
void cuda_kernel(
    const float* psum,
    const int* y0s,
    const int* y1s,
    int nx,
    int totalN,
    float Stotal,
    float* bestCost,
    int* bestX0,
    int* bestX1
) {
    int k = blockIdx.x;
    int y0 = y0s[k];
    int y1 = y1s[k];
    int h  = y1 - y0;

    extern __shared__ float col[];
    int W = nx + 1;

    // Compute column sums
    for (int x = threadIdx.x; x < nx; x += blockDim.x) {
        col[x] =
            psum[y1 * W + (x+1)] - psum[y0 * W + (x+1)]
          - psum[y1 * W + x]     + psum[y0 * W + x];
    }
    __syncthreads();

    float localBest = FLT_MAX;
    int bx0 = 0, bx1 = 1;

    // Each thread scans different x0
    for (int x0 = threadIdx.x; x0 < nx; x0 += blockDim.x) {
        float Sin = 0.0f;

        for (int x1 = x0 + 1; x1 <= nx; ++x1) {
            Sin += col[x1 - 1];

            int Nin  = h * (x1 - x0);
            int Nout = totalN - Nin;
            float Sout = Stotal - Sin;

            float cost = Sin - (Sin * Sin) / Nin;
            if (Nout > 0)
                cost += Sout - (Sout * Sout) / Nout;

            if (cost < localBest) {
                localBest = cost;
                bx0 = x0;
                bx1 = x1;
            }
        }
    }

    // Block reduction
    __shared__ float scost[256];
    __shared__ int sx0[256], sx1[256];

    int tid = threadIdx.x;
    scost[tid] = localBest;
    sx0[tid] = bx0;
    sx1[tid] = bx1;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && scost[tid + s] < scost[tid]) {
            scost[tid] = scost[tid + s];
            sx0[tid] = sx0[tid + s];
            sx1[tid] = sx1[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        bestCost[k] = scost[0];
        bestX0[k] = sx0[0];
        bestX1[k] = sx1[0];
    }
}

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float* data) {
    const int W = nx + 1;
    const int H = ny + 1;

    // Prefix sum (single channel)
    std::vector<float> psum(W * H, 0.0f);
    auto S = [&](int y, int x) -> float& {
        return psum[y * W + x];
    };

    for (int y = 1; y <= ny; ++y)
        for (int x = 1; x <= nx; ++x)
            S(y,x) = data[3*(x-1) + 3*nx*(y-1)]
                   + S(y-1,x) + S(y,x-1) - S(y-1,x-1);

    float Stotal = S(ny, nx);
    int totalN = nx * ny;

    // Flatten y-pairs
    std::vector<int> y0s, y1s;
    for (int y0 = 0; y0 < ny; ++y0)
        for (int y1 = y0 + 1; y1 <= ny; ++y1) {
            y0s.push_back(y0);
            y1s.push_back(y1);
        }

    int K = y0s.size();

    // GPU buffers
    float *d_psum, *d_bestCost;
    int *d_y0s, *d_y1s, *d_x0, *d_x1;

    cudaMalloc(&d_psum, psum.size() * sizeof(float));
    cudaMalloc(&d_y0s, K * sizeof(int));
    cudaMalloc(&d_y1s, K * sizeof(int));
    cudaMalloc(&d_bestCost, K * sizeof(float));
    cudaMalloc(&d_x0, K * sizeof(int));
    cudaMalloc(&d_x1, K * sizeof(int));

    cudaMemcpy(d_psum, psum.data(), psum.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y0s, y0s.data(), K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y1s, y1s.data(), K * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    cuda_kernel<<<K, threads, nx * sizeof(float)>>>(
        d_psum, d_y0s, d_y1s,
        nx, totalN, Stotal,
        d_bestCost, d_x0, d_x1
    );

    // Copy results back
    std::vector<float> bestCost(K);
    std::vector<int> bestX0(K), bestX1(K);

    cudaMemcpy(bestCost.data(), d_bestCost, K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bestX0.data(), d_x0, K * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(bestX1.data(), d_x1, K * sizeof(int), cudaMemcpyDeviceToHost);

    // Global best
    int bestIdx = 0;
    for (int i = 1; i < K; ++i)
        if (bestCost[i] < bestCost[bestIdx])
            bestIdx = i;

    Result res{};
    res.y0 = y0s[bestIdx];
    res.y1 = y1s[bestIdx];
    res.x0 = bestX0[bestIdx];
    res.x1 = bestX1[bestIdx];

    float Sin =
        S(res.y1,res.x1) - S(res.y0,res.x1)
      - S(res.y1,res.x0) + S(res.y0,res.x0);

    int Nin = (res.y1 - res.y0) * (res.x1 - res.x0);
    float inner = Sin / Nin;
    float outer = (Stotal - Sin) / (totalN - Nin);

    for (int c = 0; c < 3; ++c) {
        res.inner[c] = inner;
        res.outer[c] = outer;
    }

    cudaFree(d_psum);
    cudaFree(d_y0s);
    cudaFree(d_y1s);
    cudaFree(d_bestCost);
    cudaFree(d_x0);
    cudaFree(d_x1);

    return res;
}

