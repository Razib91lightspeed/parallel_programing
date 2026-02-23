#include <cmath>
#include <vector>
#include <algorithm>

void correlate(int ny, int nx, const float* data, float* result) {
    // Step 1: Normalize rows - parallelize over rows
    std::vector<double> normalized(ny * nx);
    
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int y = 0; y < ny; ++y) {
            // Compute mean of row y
            double mean = 0.0;
            for (int x = 0; x < nx; ++x) {
                mean += static_cast<double>(data[x + y * nx]);
            }
            mean /= nx;
            
            // Center the row and compute sum of squares
            double sum_sq = 0.0;
            for (int x = 0; x < nx; ++x) {
                double val = static_cast<double>(data[x + y * nx]) - mean;
                normalized[x + y * nx] = val;
                sum_sq += val * val;
            }
            
            // Normalize to unit length
            double scale = std::sqrt(sum_sq);
            if (scale > 0.0) {
                for (int x = 0; x < nx; ++x) {
                    normalized[x + y * nx] /= scale;
                }
            }
        }
    }
    
    // Step 2: Compute Y = X * X^T for upper triangle (j <= i)
    // Parallelize over the output rows - each row i can be computed independently
    
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < ny; ++i) {
        // For row i, compute columns j = 0 to i
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            
            // Dot product of row i and row j
            // Unroll by 4 for some instruction-level parallelism within each thread
            int k = 0;
            for (; k + 3 < nx; k += 4) {
                sum += normalized[(k+0) + i * nx] * normalized[(k+0) + j * nx];
                sum += normalized[(k+1) + i * nx] * normalized[(k+1) + j * nx];
                sum += normalized[(k+2) + i * nx] * normalized[(k+2) + j * nx];
                sum += normalized[(k+3) + i * nx] * normalized[(k+3) + j * nx];
            }
            for (; k < nx; ++k) {
                sum += normalized[k + i * nx] * normalized[k + j * nx];
            }
            
            result[i + j * ny] = static_cast<float>(sum);
        }
    }
}
