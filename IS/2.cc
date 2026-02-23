#include <vector>
#include <cfloat>
#include <cmath>

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
    // prefix sums: size (ny+1) x (nx+1)
    std::vector<double> sum((ny + 1) * (nx + 1) * 3, 0.0);
    std::vector<double> sqsum((ny + 1) * (nx + 1), 0.0);

    auto S = [&](int y, int x, int c) -> double& {
        return sum[c + 3 * x + 3 * (nx + 1) * y];
    };
    auto SS = [&](int y, int x) -> double& {
        return sqsum[x + (nx + 1) * y];
    };

    // build prefix sums
    for (int y = 1; y <= ny; ++y) {
        for (int x = 1; x <= nx; ++x) {
            for (int c = 0; c < 3; ++c) {
                double v = data[idx(y - 1, x - 1, c, nx)];
                S(y, x, c) = v
                           + S(y - 1, x, c)
                           + S(y, x - 1, c)
                           - S(y - 1, x - 1, c);
            }
            double ss = 0.0;
            for (int c = 0; c < 3; ++c) {
                double v = data[idx(y - 1, x - 1, c, nx)];
                ss += v * v;
            }
            SS(y, x) = ss
                     + SS(y - 1, x)
                     + SS(y, x - 1)
                     - SS(y - 1, x - 1);
        }
    }

    double totalSum[3];
    for (int c = 0; c < 3; ++c)
        totalSum[c] = S(ny, nx, c);
    double totalSq = SS(ny, nx);
    int totalN = nx * ny;

    double bestCost = DBL_MAX;
    Result best{0, 0, 1, 1, {0, 0, 0}, {0, 0, 0}};

    // try all rectangles
    for (int y0 = 0; y0 < ny; ++y0)
        for (int y1 = y0 + 1; y1 <= ny; ++y1)
            for (int x0 = 0; x0 < nx; ++x0)
                for (int x1 = x0 + 1; x1 <= nx; ++x1) {

                    int Nin = (y1 - y0) * (x1 - x0);
                    int Nout = totalN - Nin;

                    double Sin[3];
                    for (int c = 0; c < 3; ++c) {
                        Sin[c] = S(y1, x1, c)
                               - S(y0, x1, c)
                               - S(y1, x0, c)
                               + S(y0, x0, c);
                    }

                    double SSin = SS(y1, x1)
                                - SS(y0, x1)
                                - SS(y1, x0)
                                + SS(y0, x0);

                    double Sout[3];
                    for (int c = 0; c < 3; ++c)
                        Sout[c] = totalSum[c] - Sin[c];
                    double SSout = totalSq - SSin;

                    double cost = 0.0;
                    double normIn = 0.0, normOut = 0.0;
                    for (int c = 0; c < 3; ++c) {
                        normIn  += Sin[c]  * Sin[c];
                        normOut += Sout[c] * Sout[c];
                    }

                    cost += SSin  - normIn  / Nin;
                    if (Nout > 0)
                        cost += SSout - normOut / Nout;

                    if (cost < bestCost) {
                        bestCost = cost;
                        best.y0 = y0;
                        best.y1 = y1;
                        best.x0 = x0;
                        best.x1 = x1;

                        for (int c = 0; c < 3; ++c) {
                            best.inner[c] = Sin[c] / Nin;
                            best.outer[c] = (Nout > 0) ? (Sout[c] / Nout) : 0.0f;
                        }
                    }
                }

    return best;
}
