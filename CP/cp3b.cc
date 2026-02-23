#include "cp.h"
#include <cstdlib>
#include <cmath>
#include "vector.h"


constexpr int VEC_WIDTH  = 8;
constexpr int BLOCK_SIZE = 10;
float row_mean(
    const int& nx,
    const float8_t* dataPacked,
    const int& rowId,
    const int& vecCount,
    const int& strideX
) {
    float8_t accum[BLOCK_SIZE];
    const float8_t zero = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

    for (int i = 0; i < BLOCK_SIZE; ++i)
        accum[i] = zero;

    const float8_t* rowPtr = dataPacked + (size_t)rowId * strideX;

    for (int v = 0; v < vecCount; v += BLOCK_SIZE)
        for (int i = 0; i < BLOCK_SIZE; ++i)
            accum[i] += rowPtr[v + i];

    float sum = 0.f;
    for (int i = 0; i < BLOCK_SIZE; ++i)
        for (int j = 0; j < VEC_WIDTH; ++j)
            sum += accum[i][j];

    return sum / (float)nx;
}

float row_norm(
    const int& rowId,
    const float8_t* dataPacked,
    const int& vecCount,
    const int& strideX
) {
    float8_t accum[BLOCK_SIZE];
    for (int i = 0; i < BLOCK_SIZE; ++i)
        for (int j = 0; j < VEC_WIDTH; ++j)
            accum[i][j] = 0.f;

    const float8_t* rowPtr = dataPacked + (size_t)rowId * strideX;

    for (int v = 0; v < vecCount; v += BLOCK_SIZE)
        for (int i = 0; i < BLOCK_SIZE; ++i)
            accum[i] += rowPtr[v + i] * rowPtr[v + i];

    float sum = 0.f;
    for (int i = 0; i < BLOCK_SIZE; ++i)
        for (int j = 0; j < VEC_WIDTH; ++j)
            sum += accum[i][j];

    return sqrt(sum);
}
void block_dot(
    const int& colBlock,
    const float8_t* dataPacked,
    const int& strideX,
    float8_t* blockAccum,
    const int& runOffset,
    const float8_t* rowCache
) {
    const float8_t zero = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

    for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; ++i)
        blockAccum[i] = zero;

    for (int r = 0; r < BLOCK_SIZE; ++r)
        for (int c = 0; c < BLOCK_SIZE; ++c)
            for (int k = 0; k < BLOCK_SIZE; ++k)
                blockAccum[c + r * BLOCK_SIZE] +=
                    rowCache[k + r * BLOCK_SIZE] *
                    dataPacked[k + runOffset + (c + colBlock) * strideX];
}

void correlate(int ny, int nx, const float* data, float* result)
{
    const int vecCount     = (nx + VEC_WIDTH - 1) / VEC_WIDTH;
    const int fullVecCount = vecCount - (nx % VEC_WIDTH != 0);
    const int strideX      = ((vecCount + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    const int strideY      = ((ny + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    float8_t* dataPacked =
        float8_alloc((size_t)strideY * strideX);

    #pragma omp parallel for
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < strideX; ++x)
            for (int d = 0; d < VEC_WIDTH; ++d) {
                int col = d + x * VEC_WIDTH;
                dataPacked[x + strideX * y][d] =
                    (col < nx) ? data[y * nx + col] : 0.0f;
            }
    const float8_t zero = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    for (int y = ny; y < strideY; ++y)
        for (int x = 0; x < strideX; ++x)
            dataPacked[x + strideX * y] = zero;
    #pragma omp parallel for
    for (int y = 0; y < ny; ++y) {
        float mean = row_mean(nx, dataPacked, y, vecCount, strideX);
        const float8_t meanVec =
            {mean,mean,mean,mean,mean,mean,mean,mean};

        for (int v = 0; v < fullVecCount; ++v)
            dataPacked[v + strideX * y] -= meanVec;

        for (int d = 0; d < nx % VEC_WIDTH; ++d)
            dataPacked[(y + 1) * strideX - (strideX - vecCount) - 1][d] -= mean;
    }
    #pragma omp parallel for
    for (int y = 0; y < ny; ++y) {
        float norm = row_norm(y, dataPacked, vecCount, strideX);
        float inv  = 1.0f / norm;
        const float8_t normVec = {inv,inv,inv,inv,inv,inv,inv,inv};

        for (int v = 0; v < fullVecCount; ++v)
            dataPacked[v + strideX * y] *= normVec;

        for (int d = 0; d < nx % VEC_WIDTH; ++d)
            dataPacked[(y + 1) * strideX - (strideX - vecCount) - 1][d] *= inv;
    }
    float* corr =
        (float*)malloc(sizeof(float) * (size_t)strideY * strideY);

    #pragma omp parallel for
    for (int i = 0; i < strideY * strideY; ++i)
        corr[i] = 0.0f;
    #pragma omp parallel
    {
        for (int run = 0; run < strideX; run += BLOCK_SIZE) {

            #pragma omp for schedule(static,1)
            for (int y = 0; y < strideY; y += BLOCK_SIZE) {

                float8_t rowCache[BLOCK_SIZE * BLOCK_SIZE];
                for (int r = 0; r < BLOCK_SIZE; ++r)
                    for (int c = 0; c < BLOCK_SIZE; ++c)
                        rowCache[c + r * BLOCK_SIZE] =
                            dataPacked[run + c + (y + r) * strideX];

                for (int x = y; x < strideY; x += BLOCK_SIZE) {

                    float8_t blockSum[BLOCK_SIZE * BLOCK_SIZE];
                    block_dot(x, dataPacked, strideX,
                              blockSum, run, rowCache);

                    for (int i = 0; i < BLOCK_SIZE; ++i)
                        for (int j = 0; j < BLOCK_SIZE; ++j) {
                            float s = 0.0f;
                            for (int k = 0; k < VEC_WIDTH; ++k)
                                s += blockSum[j + i * BLOCK_SIZE][k];
                            corr[j + x + (y + i) * strideY] += s;
                        }
                }
            }
        }
    }
    #pragma omp parallel for
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < ny; ++x)
            result[x + y * ny] = corr[x + y * strideY];

    free(dataPacked);
    free(corr);
}
