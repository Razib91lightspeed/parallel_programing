#include <cmath>
#include <vector>
#include <algorithm>
#include <immintrin.h>

void correlate(int ny, int nx, const float* data, float* result) {
    const int VEC_SIZE = 4; 
    std::vector<double> normalized(ny * nx);

    for (int y = 0; y < ny; ++y) {
        double mean = 0.0;
        for (int x = 0; x < nx; ++x) {
            mean += static_cast<double>(data[x + y * nx]);
        }
        mean /= nx;
        for (int x = 0; x < nx; ++x) {
            normalized[x + y * nx] = static_cast<double>(data[x + y * nx]) - mean;
        }
        __m256d sum_sq_vec = _mm256_setzero_pd();
        int x = 0;
        for (; x + VEC_SIZE - 1 < nx; x += VEC_SIZE) {
            __m256d val = _mm256_loadu_pd(&normalized[x + y * nx]);
            sum_sq_vec = _mm256_fmadd_pd(val, val, sum_sq_vec);
        }
        double sum_sq = 0.0;
        double temp[4];
        _mm256_storeu_pd(temp, sum_sq_vec);
        for (int i = 0; i < 4; ++i) sum_sq += temp[i];
        for (; x < nx; ++x) {
            sum_sq += normalized[x + y * nx] * normalized[x + y * nx];
        }
        double scale = std::sqrt(sum_sq);
        if (scale > 0.0) {
            __m256d scale_vec = _mm256_set1_pd(scale);
            x = 0;
            for (; x + VEC_SIZE - 1 < nx; x += VEC_SIZE) {
                __m256d val = _mm256_loadu_pd(&normalized[x + y * nx]);
                val = _mm256_div_pd(val, scale_vec);
                _mm256_storeu_pd(&normalized[x + y * nx], val);
            }
            for (; x < nx; ++x) {
                normalized[x + y * nx] /= scale;
            }
        }
    }
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j <= i; ++j) {
            __m256d sum_vec = _mm256_setzero_pd();

            int k = 0;
            for (; k + VEC_SIZE - 1 < nx; k += VEC_SIZE) {
                __m256d a = _mm256_loadu_pd(&normalized[k + i * nx]);
                __m256d b = _mm256_loadu_pd(&normalized[k + j * nx]);
                sum_vec = _mm256_fmadd_pd(a, b, sum_vec);
            }
            double sum = 0.0;
            double temp[4];
            _mm256_storeu_pd(temp, sum_vec);
            for (int i = 0; i < 4; ++i) sum += temp[i];

            for (; k < nx; ++k) {
                sum += normalized[k + i * nx] * normalized[k + j * nx];
            }

            result[i + j * ny] = static_cast<float>(sum);
        }
    }
}
