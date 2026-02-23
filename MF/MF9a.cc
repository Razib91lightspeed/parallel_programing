#include <vector>
#include <algorithm>
#include <cstdint>
#include <omp.h>


static inline __attribute__((always_inline, hot))
int find_kth_bit(const uint64_t* bits, int nwords, int k)
{
    int base = 0;

    #pragma GCC unroll 4
    for (int w = 0; w < nwords; ++w) {
        uint64_t x = bits[w];
        int cnt = __builtin_popcountll(x);

        if (__builtin_expect(k < cnt, 0)) {
            while (true) {
                int tz = __builtin_ctzll(x);
                if (k == 0) return base + tz;
                x &= x - 1;
                --k;
            }
        }

        k -= cnt;
        base += 64;
    }
    return 0;
}

void mf(int ny, int nx, int hy, int hx,
        const float* __restrict__ in,
        float* __restrict__ out)
{
    constexpr int B = 108;

    int nbx = (nx + B - 1) / B;
    int nby = (ny + B - 1) / B;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int by = 0; by < nby; ++by)
    for (int bx = 0; bx < nbx; ++bx) {


        int x0 = std::max(0, bx * B - hx);
        int x1 = std::min(nx, (bx + 1) * B + hx);
        int y0 = std::max(0, by * B - hy);
        int y1 = std::min(ny, (by + 1) * B + hy);

        int W = x1 - x0;
        int H = y1 - y0;
        int N = W * H;

        std::vector<float> values(N);
        std::vector<int> order(N), rank(N);
        std::vector<float> rank_to_value(N);


        for (int j = 0; j < H; ++j)
            for (int i = 0; i < W; ++i)
                values[i + j * W] =
                    in[(x0 + i) + (y0 + j) * nx];


        for (int i = 0; i < N; ++i) order[i] = i;

        std::sort(order.begin(), order.end(),
                  [&](int a, int b) {
                      return values[a] < values[b];
                  });

        for (int i = 0; i < N; ++i) {
            rank[order[i]] = i;
            rank_to_value[i] = values[order[i]];
        }

        const float* r2v = rank_to_value.data();


        int ox0 = bx * B;
        int ox1 = std::min((bx + 1) * B, nx);
        int oy0 = by * B;
        int oy1 = std::min((by + 1) * B, ny);

        int nwords = (N + 63) / 64;
        std::vector<uint64_t> bitset(nwords);

        for (int y = oy0; y < oy1; ++y) {

            int wy0 = std::max(y - hy, y0);
            int wy1 = std::min(y + hy, y1 - 1);
            int wy_height = wy1 - wy0 + 1;

            std::fill(bitset.begin(), bitset.end(), 0);
            uint64_t* bits = bitset.data();


            for (int j = wy0; j <= wy1; ++j) {
                int* rrow = &rank[(j - y0) * W];
                for (int i = ox0 - hx; i <= ox0 + hx; ++i) {
                    if (i < x0 || i >= x1) continue;
                    int r = rrow[i - x0];
                    bits[r >> 6] |= 1ULL << (r & 63);
                }
            }


            for (int x = ox0; x < ox1; ++x) {

                int wx0 = std::max(x - hx, x0);
                int wx1 = std::min(x + hx, x1 - 1);
                int window_size =
                    (wx1 - wx0 + 1) * wy_height;

                if (__builtin_expect(window_size & 1, 1)) {
                    out[x + y * nx] =
                        r2v[find_kth_bit(
                            bits, nwords,
                            window_size / 2)];
                } else {
                    int m1 = find_kth_bit(
                        bits, nwords,
                        window_size / 2 - 1);
                    int m2 = find_kth_bit(
                        bits, nwords,
                        window_size / 2);
                    out[x + y * nx] =
                        0.5f * (r2v[m1] + r2v[m2]);
                }

                
                int x_rem = x - hx;
                int x_add = x + hx + 1;

                for (int j = wy0; j <= wy1; ++j) {
                    int* rrow = &rank[(j - y0) * W];
                    if (x_rem >= x0 && x_rem < x1) {
                        int r = rrow[x_rem - x0];
                        bits[r >> 6] &=
                            ~(1ULL << (r & 63));
                    }
                    if (x_add >= x0 && x_add < x1) {
                        int r = rrow[x_add - x0];
                        bits[r >> 6] |=
                            1ULL << (r & 63);
                    }
                }
            }
        }
    }
}
