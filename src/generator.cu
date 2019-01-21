#include "generator.hpp"
#include "cuda_helper.hpp"

struct ViewInfo {
    Color *image;
    int length, stride;
    Complex<double> translation;
    double scale;
};

__device__ bool mandelbrot(Complex<double> pos) {
    Complex<double> z;
    for (int i = 0; i < 50; i++) {
        z = z*z + pos;
    }
    return z.absSqr() < 4;
}

__global__ void renderImageKernel(ViewInfo info) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= info.length) {
        return;
    }

    Complex<double> pos{double(index % info.stride), double(index / info.stride)};
    pos = pos*info.scale + info.translation;
    info.image[index] = (mandelbrot(pos) ? Color(255, 255, 255) : Color(0, 0, 0));
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
