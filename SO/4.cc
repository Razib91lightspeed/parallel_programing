#include <algorithm>
#include <cstring>
#include <omp.h>
#include <vector>

typedef unsigned long long data_t;


static const int BLOCK_SIZE = 1 << 19;

static inline void merge(
    const data_t* __restrict__ src,
    data_t* __restrict__ dst,
    int left,
    int mid,
    int right
) {
    int i = left, j = mid, k = left;

    while (i + 1 < mid && j + 1 < right) {
        data_t a = src[i];
        data_t b = src[j];

        if (a <= b) {
            dst[k++] = a;
            ++i;
        } else {
            dst[k++] = b;
            ++j;
        }

        a = src[i];
        b = src[j];

        if (a <= b) {
            dst[k++] = a;
            ++i;
        } else {
            dst[k++] = b;
            ++j;
        }
    }
    while (i < mid && j < right)
        dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];

    while (i < mid)   dst[k++] = src[i++];
    while (j < right) dst[k++] = src[j++];
}

void psort(int n, data_t* data) {
    std::vector<data_t> buffer(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        int end = i + BLOCK_SIZE;
        if (end > n) end = n;
        std::sort(data + i, data + end);
    }
    data_t* src = data;
    data_t* dst = buffer.data();
    for (int width = BLOCK_SIZE; width < n; width <<= 1) {

        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int i = 0; i < n; i += 2 * width) {
                int left = i;

                int mid = left + width;
                if (mid > n) mid = n;

                int right = mid + width;
                if (right > n) right = n;

                if (mid < right) {
                    merge(src, dst, left, mid, right);
                } else {
                    std::memcpy(dst + left,
                                src + left,
                                (right - left) * sizeof(data_t));
                }
            }
        }
        std::swap(src, dst);
    }
    if (src != data) {
        std::memcpy(data, src, n * sizeof(data_t));
    }
}
