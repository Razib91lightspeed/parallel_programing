#include <algorithm>
#include <omp.h>

typedef unsigned long long data_t;

static const int MAX_TASK_DEPTH = 8;

void quicksort_parallel(data_t* data, int left, int right, int depth) {
    if (right - left <= 1)
        return;
    data_t a = data[left];
    data_t b = data[left + (right - left) / 2];
    data_t c = data[right - 1];
    data_t pivot = std::max(std::min(a, b),
                            std::min(std::max(a, b), c));

    int i = left;
    int j = right - 1;
    while (i <= j) {
        while (data[i] < pivot) ++i;
        while (data[j] > pivot) --j;
        if (i <= j) {
            std::swap(data[i], data[j]);
            ++i;
            --j;
        }
    }
    if (depth < MAX_TASK_DEPTH) {
        #pragma omp task
        quicksort_parallel(data, left, j + 1, depth + 1);

        quicksort_parallel(data, i, right, depth + 1);
    } else {
        quicksort_parallel(data, left, j + 1, depth);
        quicksort_parallel(data, i, right, depth);
    }
}

void psort(int n, data_t* data) {
    if (n <= 1) return;
    #pragma omp parallel
    {
        #pragma omp single
        {
            quicksort_parallel(data, 0, n, 0);
        }
    }
}
