# Pearson Correlation Calculator

## Overview
This C++ code computes the Pearson correlation coefficients between rows of a data matrix using double-precision arithmetic for improved numerical stability.

## Features
- **Row-wise normalization**: Centers each row to zero mean and scales to unit Euclidean length
- **Double precision**: Uses `double` internally to reduce floating-point errors
- **Efficient computation**: Computes only the upper triangular part of the symmetric correlation matrix

## Algorithm
1. **Normalization**: For each row:
   - Subtract the mean
   - Divide by the row's Euclidean norm
2. **Correlation**: Compute dot products between normalized rows:
   `corr(i,j) = dot(norm_row_i, norm_row_j)`

## Usage
Call `correlate(ny, nx, data, result)` where:
- `ny`: number of rows
- `nx`: number of columns
- `data`: input matrix (row-major, size `ny*nx`)
- `result`: output matrix (size `ny*ny`, upper triangle stored in row-major order)

## Output Format
The symmetric correlation matrix is stored in `result` with elements `result[i + j*ny]` for `j <= i`.

# End