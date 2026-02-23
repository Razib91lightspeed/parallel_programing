#include "cp.h"
#include <cstdlib>
#include <cmath>
#include <algorithm>

constexpr int VEC_WIDTH = 4;
constexpr int BLOCK_Y = 64;
constexpr int BLOCK_X = 64;

static inline size_t align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

void correlate(int ny, int nx, const float* data, float* result)
{
    const int nxAligned = ((nx + 3) / 4) * 4;
    size_t dataT_size = align_size(sizeof(double) * nxAligned * ny, 64);
    double* dataT = (double*)aligned_alloc(64, dataT_size);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; ++y) {
        const float* srcRow = data + y * nx;
        for (int x = 0; x < nx; ++x) {
            dataT[x * ny + y] = (double)srcRow[x];
        }
        for (int x = nx; x < nxAligned; ++x) {
            dataT[x * ny + y] = 0.0;
        }
    }
    size_t means_size = align_size(sizeof(double) * ny, 64);
    double* means = (double*)aligned_alloc(64, means_size);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; ++y) {
        double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        int x = 0;
        for (; x + 3 < nx; x += 4) {
            sum0 += dataT[x * ny + y];
            sum1 += dataT[(x + 1) * ny + y];
            sum2 += dataT[(x + 2) * ny + y];
            sum3 += dataT[(x + 3) * ny + y];
        }
        double sum = sum0 + sum1 + sum2 + sum3;
        for (; x < nx; ++x) {
            sum += dataT[x * ny + y];
        }
        means[y] = sum / nx;
    }
    #pragma omp parallel for schedule(static)
    for (int x = 0; x < nx; ++x) {
        double* col = dataT + x * ny;
        for (int y = 0; y < ny; ++y) {
            col[y] -= means[y];
        }
    }
    free(means);
    double* norms = (double*)aligned_alloc(64, means_size); // Same size as means
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; ++y) {
        double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        int x = 0;
        for (; x + 3 < nx; x += 4) {
            double v0 = dataT[x * ny + y];
            double v1 = dataT[(x + 1) * ny + y];
            double v2 = dataT[(x + 2) * ny + y];
            double v3 = dataT[(x + 3) * ny + y];
            sum0 += v0 * v0;
            sum1 += v1 * v1;
            sum2 += v2 * v2;
            sum3 += v3 * v3;
        }
        double sum = sum0 + sum1 + sum2 + sum3;
        for (; x < nx; ++x) {
            double v = dataT[x * ny + y];
            sum += v * v;
        }
        norms[y] = sqrt(sum);
    }
    #pragma omp parallel for schedule(static)
    for (int x = 0; x < nx; ++x) {
        double* col = dataT + x * ny;
        for (int y = 0; y < ny; ++y) {
            col[y] /= norms[y];
        }
    }
    free(norms);
    const int numBlocks = (ny + BLOCK_Y - 1) / BLOCK_Y;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int blockIdx = 0; blockIdx < numBlocks * numBlocks; ++blockIdx) {
        int bi = blockIdx / numBlocks;
        int bj = blockIdx % numBlocks;

        int ii = bi * BLOCK_Y;
        int jj = bj * BLOCK_X;

        if (ii >= ny || jj >= ny) continue;
        if (jj > ii) continue;

        int iMax = std::min(ii + BLOCK_Y, ny);
        int jMax = std::min(jj + BLOCK_X, ny);

        double local[BLOCK_Y * BLOCK_X];
        for (int i = ii; i < iMax; ++i) {
            int jStart = jj;
            int jEnd = std::min(jMax, i + 1);
            if (jStart > i) continue;
            for (int j = jStart; j < jEnd; ++j) {
                local[(i - ii) * BLOCK_X + (j - jj)] = 0.0;
            }
        }

        for (int k = 0; k < nx; ++k) {
            const double* colK = dataT + k * ny;
            for (int i = ii; i < iMax; ++i) {
                double a = colK[i];
                int jStart = jj;
                int jEnd = std::min(jMax, i + 1);
                if (jStart > i) continue;
                #pragma omp simd
                for (int j = jStart; j < jEnd; ++j) {
                    local[(i - ii) * BLOCK_X + (j - jj)] += a * colK[j];
                }
            }
        }

        for (int i = ii; i < iMax; ++i) {
            int jStart = jj;
            int jEnd = std::min(jMax, i + 1);
            if (jStart > i) continue;
            for (int j = jStart; j < jEnd; ++j) {
                result[i + j * ny] = (float)local[(i - ii) * BLOCK_X + (j - jj)];
            }
        }
    }

    free(dataT);
}
