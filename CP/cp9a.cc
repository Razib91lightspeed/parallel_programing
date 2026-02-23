#include <vector>
#include <cmath>
#include <immintrin.h>

void correlate(int ny, int nx, const float *data, float *result)
{
    if (ny <= 0 || nx <= 0) return;

    std::vector<float> norm((size_t)ny * nx);

    // ---------- Normalize rows ----------
    #pragma omp parallel for
    for (int i = 0; i < ny; ++i) {

        const float* row = data + (size_t)i * nx;
        float* out = norm.data() + (size_t)i * nx;

        double sum = 0.0;
        for (int k = 0; k < nx; ++k) sum += row[k];
        double mean = sum / nx;

        double sq = 0.0;
        for (int k = 0; k < nx; ++k) {
            double v = row[k] - mean;
            out[k] = (float)v;
            sq += v * v;
        }

        float inv = (sq > 1e-20) ? (float)(1.0 / std::sqrt(sq)) : 0.0f;

        for (int k = 0; k < nx; ++k)
            out[k] *= inv;
    }

    // ---------- Correlation (SIMD) ----------
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < ny; ++i) {

        const float* row_i = norm.data() + (size_t)i * nx;

        for (int j = 0; j <= i; ++j) {

            const float* row_j = norm.data() + (size_t)j * nx;

            __m256 acc = _mm256_setzero_ps();

            int k = 0;
            for (; k + 8 <= nx; k += 8) {

                __m256 a = _mm256_loadu_ps(row_i + k);
                __m256 b = _mm256_loadu_ps(row_j + k);

                acc = _mm256_fmadd_ps(a, b, acc);
            }

            float sum = 0.0f;
            float tmp[8];
            _mm256_storeu_ps(tmp, acc);

            for (int t = 0; t < 8; ++t)
                sum += tmp[t];

            // tail
            for (; k < nx; ++k)
                sum += row_i[k] * row_j[k];

            result[i + (size_t)j * ny] = sum;
        }
    }
}
