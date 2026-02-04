/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <cmath>
#include <vector>

void correlate(int ny, int nx, const float *data, float *result)
{
    // Step 1: Normalize the input rows
    // Create a double-precision matrix for the normalized data
    std::vector<double> normalized(ny * nx);

    // For each row, compute mean and subtract it, then normalize to unit length
    for (int y = 0; y < ny; ++y)
    {
        // Compute mean of row y using double precision
        double mean = 0.0;
        for (int x = 0; x < nx; ++x)
        {
            mean += static_cast<double>(data[x + y * nx]);
        }
        mean /= nx;

        // Subtract mean and compute sum of squares
        double sum_sq = 0.0;
        for (int x = 0; x < nx; ++x)
        {
            double val = static_cast<double>(data[x + y * nx]) - mean;
            normalized[x + y * nx] = val;
            sum_sq += val * val;
        }

        // Normalize so that sum of squares is 1
        double scale = std::sqrt(sum_sq);
        if (scale > 0.0)
        {
            for (int x = 0; x < nx; ++x)
            {
                normalized[x + y * nx] /= scale;
            }
        }
    }

    // Step 2: Compute matrix product Y = X * X^T
    // Store in result[i + j*ny] for the upper triangle (j <= i)
    for (int i = 0; i < ny; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < nx; ++k)
            {
                sum += normalized[k + i * nx] * normalized[k + j * nx];
            }
            // Store result as float (as required by the interface)
            result[i + j * ny] = static_cast<float>(sum);
        }
    }
}
