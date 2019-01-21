#include "generator.hpp"
#include "cuda_helper.hpp"

struct ViewInfo {
    Color *image;
    int length, stride;
    Complex<double> translation;
    double scale;
};

__device__ int mandelbrot(Complex<double> pos, int maxIters) {
    Complex<double> z;
    int iters = 0;

    while (iters < maxIters && z.absSqr() < 4) {
        z = z.sqr() + pos;
        iters++;
    }

    return iters;
}

__device__ Color getColor(double i) {
    int tmp = int(i * 255 * 5) % 256;
    return Color(tmp, tmp, tmp);
}

__global__ void renderImageKernel(ViewInfo info) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= info.length) {
        return;
    }

    Complex<double> pos{double(index % info.stride), double(index / info.stride)};
    pos = pos*info.scale + info.translation;

    constexpr int maxIters = 128;
    int iters = mandelbrot(pos, maxIters);
    info.image[index] = (iters < maxIters ? getColor(double(iters) / maxIters) : Color(0, 0, 0));
}

void renderImage(const Viewport& view) {
    constexpr uint32_t blockSize = 1024;
    uint32_t nPixels = view.width * view.height;
    uint32_t nBlocks = (nPixels+blockSize-1) / blockSize;

    ViewInfo info;
    info.image = view.image;
    info.length = nPixels;
    info.stride = view.width;

    info.translation.x = view.center.x - view.scale;
    info.scale = (view.center.x - info.translation.x) * 2 / view.width;
    info.translation.y = view.center.y - info.scale * view.height / 2;

    renderImageKernel<<<nBlocks, blockSize>>>(info);
    gpuErrchk(cudaDeviceSynchronize());
}
