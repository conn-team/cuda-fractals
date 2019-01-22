#pragma once

#include <complex>
#include <cstdint>
#include <gmpxx.h>

#include "cuda_helper.hpp"
#include "complex.hpp"
#include "color.hpp"

struct RefPointInfo {
    Complex<double> value;
};

template<typename Fractal>
class RenderInfo {
private:
    __device__ Color getColor(int iters) {
        constexpr Color initColor = Color(57, 57, 191);
        constexpr float itersThreshold = 12.0;
        constexpr float saturation = 0.55;
        constexpr float value = 0.7;
        constexpr int hueOffset = 240;

        if (iters == maxIters) {
            return Color(0, 0, 0);
        }

        if (iters <= itersThreshold) {
            float scale = iters / itersThreshold;
            return Color(initColor.r * scale, initColor.g * scale, initColor.b * scale);
        }

        return HSVAColor((hueOffset + iters) % 360, saturation, value).toRGBA();
    }

public:
    __device__ void render() {
        int index = blockIdx.x*blockDim.x + threadIdx.x;
        int x = index % width, y = index / width;
        if (y >= height) {
            return;
        }

        double dx = x - double(width)/2, dy = y - double(height)/2;
        Complex<double> pos{dx*scale, dy*scale};

        Complex<double> cur = pos;
        double bailout = params.bailoutSqr();
        int iters = 0;

        while (iters < maxIters) {
            auto ref = referenceData[iters].value;
            if ((cur+ref).absSqr() >= bailout) {
                break;
            }

            cur = params.relativeStep(pos, cur, ref);
            iters++;
        }

        image[index] = getColor(iters);
    }

public:
    Fractal params;
    Color *image;
    RefPointInfo *referenceData;
    int maxIters, width, height;
    double scale;
};

template<typename Fractal>
__global__ static void renderImageKernel(RenderInfo<Fractal> info) {
    info.render();
}

class Viewport {
private:
    template<typename Fractal>
    std::vector<RefPointInfo> buildReferenceData(const Fractal& params) {
        std::vector<RefPointInfo> vec(maxIters);
        auto cur = center;

        for (int i = 0; i < maxIters; i++) {
            if (i > 0) {
                cur = params.step(center, cur);
            }
            vec[i].value = Complex<double>(cur.real().get_d(), cur.imag().get_d());
        }

        return vec;
    }

public:
    template<typename Fractal>
    void renderImage(const Fractal& params) {
        constexpr uint32_t blockSize = 1024;
        uint32_t nBlocks = (width*height+blockSize-1) / blockSize;

        devReferenceData.assign(buildReferenceData(params));

        RenderInfo<Fractal> info;
        info.params = params;
        info.image = devImage;
        info.referenceData = devReferenceData.data();
        info.maxIters = maxIters;
        info.width = width;
        info.height = height;
        info.scale = mpf_class(scale * 2 / width).get_d();

        renderImageKernel<<<nBlocks, blockSize>>>(info);
    }

private:
    CudaArray<RefPointInfo> devReferenceData;
public:
    Color *devImage;
    int maxIters, width, height;
    std::complex<mpf_class> center;
    mpf_class scale;
};
