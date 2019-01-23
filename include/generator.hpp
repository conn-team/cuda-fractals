#pragma once

#include <complex>
#include <cstdint>

#include "cuda_helper.hpp"
#include "complex.hpp"
#include "color.hpp"
#include "series.hpp"
#include "bignum.hpp"

struct RefPointInfo {
    Complex<double> value;
    CubicSeries<Complex<double>> series;
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

        double bailout = params.bailoutSqr();
        int iters = minIters;
        Complex<double> cur = referenceData[iters].series.eval(pos);

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
    int minIters, maxIters, width, height;
    Complex<double> refPointScreen;
    double scale;
};

template<typename Fractal>
__global__ static void renderImageKernel(RenderInfo<Fractal> info) {
    info.render();
}

Complex<double> downgradeComplex(const BigComplex& x) {
    return Complex<double>(double(x.real()), double(x.imag()));
}

class Viewport {
private:
    template<typename Fractal>
    int getIterations(const Fractal& params, BigComplex point) {
        int iters = maxIters;
        auto cur = point;

        for (int i = 0; i < maxIters; i++) {
            if (std::norm(cur) >= params.bailoutSqr()) {
                return i;
            }
            cur = params.step(point, cur);
        }

        return iters;
    }

    template<typename Fractal>
    int buildReferenceData(const Fractal& params, BigComplex point, std::vector<RefPointInfo>& out) {
        int iters = maxIters;
        auto cur = point;
        CubicSeries<BigComplex> series{BigComplex(1), BigComplex(0), BigComplex(0)};
        out.resize(maxIters);

        for (int i = 0; i < maxIters; i++) {
            if (i > 0) {
                series = params.seriesStep(series, cur);
                cur = params.step(point, cur);
            }

            out[i].value = downgradeComplex(cur);
            for (int j = 0; j < 3; j++) {
                out[i].series[j] = downgradeComplex(series[j]);
            }

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
        info.minIters = minIters;
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
    int minIters, maxIters, width, height;
    BigComplex center;
    BigFloat scale;
};
