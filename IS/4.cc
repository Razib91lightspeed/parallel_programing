#include <vector>
#include <cfloat>
#include <cmath>
#include <omp.h>

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

    // ---------------- Prefix sums ----------------
    std::vector<double> sum(H * W * 3, 0.0);
    std::vector<double> sqsum(H * W, 0.0);

    auto S = [&](int y, int x, int c) -> double& {
        return sum[c + 3 * x + 3 * W * y];
    };
    auto SS = [&](int y, int x) -> double& {
        return sqsum[x + W * y];
    };

    for (int y = 1; y <= ny; ++y) {
        for (int x = 1; x <= nx; ++x) {
            double s2 = 0.0;
            for (int c = 0; c < 3; ++c) {
                double v = data[idx(y - 1, x - 1, c, nx)];
                S(y, x, c) = v
                           + S(y-1, x, c)
                           + S(y, x-1, c)
                           - S(y-1, x-1, c);
                s2 += v * v;
            }
            SS(y, x) = s2
                     + SS(y-1, x)
                     + SS(y, x-1)
                     - SS(y-1, x-1);
        }
    }

    // ---------------- Totals ----------------
    const double TS0 = S(ny, nx, 0);
    const double TS1 = S(ny, nx, 1);
    const double TS2 = S(ny, nx, 2);
    const double TSQ = SS(ny, nx);
    const int totalN = nx * ny;

    // ---------------- Enumerate y-pairs ----------------
    std::vector<std::pair<int,int>> ypairs;
    ypairs.reserve(ny * (ny + 1) / 2);
    for (int y0 = 0; y0 < ny; ++y0)
        for (int y1 = y0 + 1; y1 <= ny; ++y1)
            ypairs.emplace_back(y0, y1);

    double globalBestCost = DBL_MAX;
    Result globalBest{0,0,1,1,{0,0,0},{0,0,0}};

    omp_set_dynamic(0);

    // ---------------- Parallel region ----------------
    #pragma omp parallel
    {
        double localBestCost = DBL_MAX;
        Result localBest = globalBest;

        // Pre-allocate column arrays
        std::vector<double> col0(nx);
        std::vector<double> col1(nx);
        std::vector<double> col2(nx);
        std::vector<double> colSq(nx);

        #pragma omp for schedule(static)
        for (size_t k = 0; k < ypairs.size(); ++k) {
            int y0 = ypairs[k].first;
            int y1 = ypairs[k].second;
            int h  = y1 - y0;

            // column prefix differences
            for (int x = 0; x < nx; ++x) {
                col0[x] = S(y1,x+1,0) - S(y0,x+1,0) - S(y1,x,0) + S(y0,x,0);
                col1[x] = S(y1,x+1,1) - S(y0,x+1,1) - S(y1,x,1) + S(y0,x,1);
                col2[x] = S(y1,x+1,2) - S(y0,x+1,2) - S(y1,x,2) + S(y0,x,2);
                colSq[x]= SS(y1,x+1)  - SS(y0,x+1)  - SS(y1,x)  + SS(y0,x);
            }

            for (int x0 = 0; x0 < nx; ++x0) {
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, ss = 0.0;

                for (int x1 = x0 + 1; x1 <= nx; ++x1) {
                    int x = x1 - 1;

                    s0 += col0[x];
                    s1 += col1[x];
                    s2 += col2[x];
                    ss += colSq[x];

                    int Nin  = h * (x1 - x0);
                    int Nout = totalN - Nin;

                    double cost = 0.0;
                    
                    if (Nin > 0) {
                        double invNin = 1.0 / Nin;
                        cost += ss - (s0*s0 + s1*s1 + s2*s2) * invNin;
                    }
                    
                    if (Nout > 0) {
                        double invNout = 1.0 / Nout;
                        double o0 = TS0 - s0;
                        double o1 = TS1 - s1;
                        double o2 = TS2 - s2;
                        cost += (TSQ - ss) - (o0*o0 + o1*o1 + o2*o2) * invNout;
                    }

                    if (cost < localBestCost) {
                        localBestCost = cost;
                        localBest.y0 = y0;
                        localBest.y1 = y1;
                        localBest.x0 = x0;
                        localBest.x1 = x1;
                        
                        if (Nin > 0) {
                            double invNin = 1.0 / Nin;
                            localBest.inner[0] = s0 * invNin;
                            localBest.inner[1] = s1 * invNin;
                            localBest.inner[2] = s2 * invNin;
                        }
                        
                        if (Nout > 0) {
                            double invNout = 1.0 / Nout;
                            localBest.outer[0] = (TS0 - s0) * invNout;
                            localBest.outer[1] = (TS1 - s1) * invNout;
                            localBest.outer[2] = (TS2 - s2) * invNout;
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            if (localBestCost < globalBestCost) {
                globalBestCost = localBestCost;
                globalBest = localBest;
            }
        }
    }

    return globalBest;
}
