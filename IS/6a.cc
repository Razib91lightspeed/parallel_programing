#include <vector>
#include <cfloat>
#include <omp.h>
#include <cstddef>

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

inline int idx(int y, int x, int c, int nx) {
    return c + 3 * x + 3 * nx * y;
}

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
    const int W = nx + 1;
    const int H = ny + 1;

    // Prefix sum of white pixels (single channel)
    std::vector<float> psum(H * W, 0.0f);

    auto S = [&](int y, int x) -> float& {
        return psum[x + W * y];
    };

    // Build prefix sum
    for (int y = 1; y <= ny; ++y) {
        for (int x = 1; x <= nx; ++x) {
            float v = data[idx(y - 1, x - 1, 0, nx)]; // 0 or 1
            S(y,x) = v + S(y-1,x) + S(y,x-1) - S(y-1,x-1);
        }
    }

    const float Stotal = S(ny, nx);
    const int totalN = nx * ny;

    // Flatten all (y0,y1) pairs
    std::vector<int> y0s, y1s;
    y0s.reserve(ny * (ny + 1) / 2);
    y1s.reserve(ny * (ny + 1) / 2);

    for (int y0 = 0; y0 < ny; ++y0)
        for (int y1 = y0 + 1; y1 <= ny; ++y1) {
            y0s.push_back(y0);
            y1s.push_back(y1);
        }

    float bestCost = FLT_MAX;
    Result best{0,0,1,1,{0,0,0},{0,0,0}};

    omp_set_dynamic(0);

    #pragma omp parallel
    {
        float localBestCost = FLT_MAX;
        Result localBest = best;

        std::vector<float> col(nx);

        #pragma omp for schedule(static)
        for (size_t k = 0; k < y0s.size(); ++k) {
            int y0 = y0s[k];
            int y1 = y1s[k];
            int h  = y1 - y0;

            // Column sums
            for (int x = 0; x < nx; ++x) {
                col[x] = S(y1, x+1) - S(y0, x+1)
                       - S(y1, x)   + S(y0, x);
            }

            // Sliding window in x
            for (int x0 = 0; x0 < nx; ++x0) {
                float Sin = 0.0f;

                for (int x1 = x0 + 1; x1 <= nx; ++x1) {
                    Sin += col[x1 - 1];

                    int Nin  = h * (x1 - x0);
                    int Nout = totalN - Nin;
                    float Sout = Stotal - Sin;

                    float cost =
                        Sin - (Sin * Sin) / Nin;

                    if (Nout > 0)
                        cost += Sout - (Sout * Sout) / Nout;

                    if (cost < localBestCost) {
                        localBestCost = cost;
                        localBest.y0 = y0;
                        localBest.y1 = y1;
                        localBest.x0 = x0;
                        localBest.x1 = x1;

                        float innerVal = Sin / Nin;
                        float outerVal = (Nout > 0) ? (Sout / Nout) : 0.0f;

                        for (int c = 0; c < 3; ++c) {
                            localBest.inner[c] = innerVal;
                            localBest.outer[c] = outerVal;
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            if (localBestCost < bestCost) {
                bestCost = localBestCost;
                best = localBest;
            }
        }
    }

    return best;
}

