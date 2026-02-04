struct Result
{
    float avg[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- horizontal position: 0 <= x0 < x1 <= nx
- vertical position: 0 <= y0 < y1 <= ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
- output: avg[c]
*/

Result calculate(int ny, int nx, const float *data,
                 int y0, int x0, int y1, int x1)
{
    (void)ny; // explicitly mark as unused

    struct Result result;
    result.avg[0] = 0.0f;
    result.avg[1] = 0.0f;
    result.avg[2] = 0.0f;

    double sum[3] = {0.0, 0.0, 0.0};

    for (int y = y0; y < y1; y++)
    {
        for (int x = x0; x < x1; x++)
        {
            int base = 3 * x + 3 * nx * y;
            sum[0] += data[base + 0];
            sum[1] += data[base + 1];
            sum[2] += data[base + 2];
        }
    }

    double count = (double)(y1 - y0) * (double)(x1 - x0);

    result.avg[0] = (float)(sum[0] / count);
    result.avg[1] = (float)(sum[1] / count);
    result.avg[2] = (float)(sum[2] / count);

    return result;
}
