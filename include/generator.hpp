#pragma once

#include <complex>
#include <cstdint>
#include <boost/multiprecision/mpfr.hpp>

#include "cuda_helper.hpp"
#include "complex.hpp"
#include "color.hpp"

using namespace boost::multiprecision;

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

        Complex<double> pos = Complex<double>(x, y) - refPointScreen;
        pos *= scale;

        Complex<double> cur = pos;
        double bailout = params.bailoutSqr();
        int iters = 0;

        while (iters < maxIters) {
            auto ref = referenceData[iters].value;
            if ((cur+ref).norm() >= bailout) {
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
    Complex<double> refPointScreen;
    double scale;
};

template<typename Fractal>
__global__ static void renderImageKernel(RenderInfo<Fractal> info) {
    info.render();
}

class Viewport {
private:
    template<typename Fractal>
    int buildReferenceData(const Fractal& params, std::complex<mpfr_float> point, std::vector<RefPointInfo>& out) {
        int iters = maxIters;
        auto cur = point;
        out.resize(maxIters);

        for (int i = 0; i < maxIters; i++) {
            if (i > 0) {
                cur = params.step(point, cur);
            }

            out[i].value = Complex<double>(double(cur.real()), double(cur.imag()));

            if (std::norm(cur) >= params.bailoutSqr()) {
                iters = min(iters, i+1);
            }
        }

        return iters;
    }

public:
    template<typename Fractal>
    void renderImage(const Fractal& params) {
        constexpr uint32_t blockSize = 1024;
        uint32_t nBlocks = (width*height+blockSize-1) / blockSize;

        RenderInfo<Fractal> info;
        info.params = params;
        info.image = devImage;
        info.maxIters = maxIters;
        info.width = width;
        info.height = height;
        info.scale = double(scale * 2 / width);

        std::vector<RefPointInfo> refData;
        buildReferenceData(params, center, refData);
        devReferenceData.assign(refData);

        info.referenceData = devReferenceData.data();
        info.refPointScreen = Complex<double>(width, height) * 0.5;

        renderImageKernel<<<nBlocks, blockSize>>>(info);
    }

private:
    CudaArray<RefPointInfo> devReferenceData;
public:
    Color *devImage;
    int maxIters, width, height;
    std::complex<mpfr_float> center;
    mpfr_float scale;
};
