#include <cmath>
#include <vector>
#include <algorithm>

void correlate(int ny, int nx, const float* data, float* result) {
    const int BLOCK_SIZE = 4;
    
    // Step 1: Normalize rows
    std::vector<double> normalized(ny * nx);
    
    for (int y_base = 0; y_base < ny; y_base += BLOCK_SIZE) {
        int y_end = std::min(y_base + BLOCK_SIZE, ny);
        
        // Compute means for this block
        double mean[BLOCK_SIZE] = {0.0};
        for (int x = 0; x < nx; ++x) {
            for (int y = y_base; y < y_end; ++y) {
                mean[y - y_base] += static_cast<double>(data[x + y * nx]);
            }
        }
        for (int i = 0; i < (y_end - y_base); ++i) {
            mean[i] /= nx;
        }
        
        // Center and compute sum of squares
        double sum_sq[BLOCK_SIZE] = {0.0};
        for (int x = 0; x < nx; ++x) {
            for (int y = y_base; y < y_end; ++y) {
                double val = static_cast<double>(data[x + y * nx]) - mean[y - y_base];
                normalized[x + y * nx] = val;
                sum_sq[y - y_base] += val * val;
            }
        }
        
        // Normalize to unit length
        double scale[BLOCK_SIZE];
        for (int i = 0; i < (y_end - y_base); ++i) {
            scale[i] = std::sqrt(sum_sq[i]);
            if (scale[i] == 0.0) scale[i] = 1.0;
        }
        
        for (int x = 0; x < nx; ++x) {
            for (int y = y_base; y < y_end; ++y) {
                normalized[x + y * nx] /= scale[y - y_base];
            }
        }
    }
    
    // Step 2: Compute Y = X * X^T for upper triangle (j <= i)
    // Process in blocks for cache efficiency and ILP
    
    for (int i = 0; i < ny; ++i) {
        for (int j_base = 0; j_base <= i; j_base += BLOCK_SIZE) {
            int j_end = std::min(j_base + BLOCK_SIZE, i + 1);
            
            // Process multiple j values for the same i (exploits ILP)
            int j = j_base;
            
            // Unroll by 4
            for (; j + 3 < j_end; j += 4) {
                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                
                int k = 0;
                // 4x unrolled inner loop
                for (; k + 3 < nx; k += 4) {
                    double a0 = normalized[(k+0) + i * nx];
                    double a1 = normalized[(k+1) + i * nx];
                    double a2 = normalized[(k+2) + i * nx];
                    double a3 = normalized[(k+3) + i * nx];
                    
                    sum0 += a0 * normalized[(k+0) + (j+0) * nx];
                    sum0 += a1 * normalized[(k+1) + (j+0) * nx];
                    sum0 += a2 * normalized[(k+2) + (j+0) * nx];
                    sum0 += a3 * normalized[(k+3) + (j+0) * nx];
                    
                    sum1 += a0 * normalized[(k+0) + (j+1) * nx];
                    sum1 += a1 * normalized[(k+1) + (j+1) * nx];
                    sum1 += a2 * normalized[(k+2) + (j+1) * nx];
                    sum1 += a3 * normalized[(k+3) + (j+1) * nx];
                    
                    sum2 += a0 * normalized[(k+0) + (j+2) * nx];
                    sum2 += a1 * normalized[(k+1) + (j+2) * nx];
                    sum2 += a2 * normalized[(k+2) + (j+2) * nx];
                    sum2 += a3 * normalized[(k+3) + (j+2) * nx];
                    
                    sum3 += a0 * normalized[(k+0) + (j+3) * nx];
                    sum3 += a1 * normalized[(k+1) + (j+3) * nx];
                    sum3 += a2 * normalized[(k+2) + (j+3) * nx];
                    sum3 += a3 * normalized[(k+3) + (j+3) * nx];
                }
                
                // Remainder
                for (; k < nx; ++k) {
                    double a = normalized[k + i * nx];
                    sum0 += a * normalized[k + (j+0) * nx];
                    sum1 += a * normalized[k + (j+1) * nx];
                    sum2 += a * normalized[k + (j+2) * nx];
                    sum3 += a * normalized[k + (j+3) * nx];
                }
                
                result[i + (j+0) * ny] = static_cast<float>(sum0);
                result[i + (j+1) * ny] = static_cast<float>(sum1);
                result[i + (j+2) * ny] = static_cast<float>(sum2);
                result[i + (j+3) * ny] = static_cast<float>(sum3);
            }
            
            // Handle remaining 1-3 j values
            for (; j < j_end; ++j) {
                double sum = 0.0;
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
}
