#include <vector>
#include <algorithm>
#include <omp.h>

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/


void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {

            // Determine window boundaries (clamped to image)
            int x0 = std::max(0, x - hx);
            int x1 = std::min(nx - 1, x + hx);
            int y0 = std::max(0, y - hy);
            int y1 = std::min(ny - 1, y + hy);

            // Thread-private window
            std::vector<float> window;
            for (int j = y0; j <= y1; ++j) {
                for (int i = x0; i <= x1; ++i) {
                    window.push_back(in[i + j * nx]);
                }
            }

            int n = window.size();
            int mid = n / 2;

            std::nth_element(window.begin(),
                             window.begin() + mid,
                             window.end());

            float median;
            if (n % 2 == 1) {
                median = window[mid];
            } else {
                float upper = window[mid];
                std::nth_element(window.begin(),
                                 window.begin() + mid - 1,
                                 window.end());
                median = 0.5f * (upper + window[mid - 1]);
            }

            out[x + y * nx] = median;
        }
    }
}
