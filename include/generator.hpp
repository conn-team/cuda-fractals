#pragma once

#include <complex>
#include <cstdint>
#include <iostream>

#include "cuda_helper.hpp"
#include "complex.hpp"
#include "color.hpp"
#include "series.hpp"
#include "bignum.hpp"

struct RefPointInfo {
    DevComplex value;
    CubicSeries<DevComplex> series;
};

template<typename Fractal>
class RenderInfo {
private:
    __device__ Color getColor(int iters) {
        if (iters == maxIters) {
            return Color(0, 0, 0);
        }

        Color colors[] = {
            {0, 0, 0},
            {200, 0, 0},
            {200, 200, 0},
            {0, 200, 200},
            {0, 0, 200},
        };

        float subIters = iters % 160 / 32.f;

        int c0 = floor(subIters - 5.0f * floor(subIters / 5.0f));
        int c1 = ceil (subIters - 5.0f * floor(subIters / 5.0f));
        float t = subIters - c0;
        c1 %= 5;

        return {
            static_cast<uint8_t>((1.0f - t) * colors[c0].r + t * colors[c1].r),
            static_cast<uint8_t>((1.0f - t) * colors[c0].g + t * colors[c1].g),
            static_cast<uint8_t>((1.0f - t) * colors[c0].b + t * colors[c1].b)
        };
    }

public:
    __both__ DevComplex screenToDelta(int x, int y) {
        return (DevComplex(x, y) - refPointScreen) * scale;
    }

    __device__ void render() {
        int index = blockIdx.x*blockDim.x + threadIdx.x;
        int x = index % width, y = index / width;
        if (y >= height) {
            return;
        }

        DevComplex pos = screenToDelta(x, y);
        int iters = minIters;
        DevComplex cur = referenceData[iters].series.eval(pos);

        while (iters+1 < approxIters) {
            auto ref = referenceData[iters].value;
            if ((cur+ref).norm() >= params.bailoutSqr()) {
                break;
            }

            cur = params.relativeStep(pos, cur, ref);
            iters++;
        }

        pos += referenceData[0].value;
        cur += referenceData[iters].value;

        while (iters < maxIters) {
            if (cur.norm() >= params.bailoutSqr()) {
                break;
            }

            cur = params.relativeStep(pos, cur, DevComplex(0, 0));
            iters++;
        }

        image[index] = getColor(iters);
    }

public:
    Fractal params;
    Color *image;
    RefPointInfo *referenceData;
    int minIters, maxIters, approxIters, width, height;
    DevComplex refPointScreen;
    double scale;
};

template<typename Fractal>
__global__ static void renderImageKernel(RenderInfo<Fractal> info) {
    info.render();
}

DevComplex downgradeComplex(const BigComplex& x) {
    return DevComplex(double(x.real()), double(x.imag()));
}

BigComplex upgradeComplex(const DevComplex& c) {
    return BigComplex(BigFloat(c.x), BigFloat(c.y));
}

class Viewport {
private:
    template<typename Fractal>
    int buildReferenceData(const Fractal& params, const BigComplex& point, std::vector<RefPointInfo>& out) {
        int iters = maxIters;
        auto cur = point;
        CubicSeries<DevComplex> series{1, 0, 0};
        out.resize(maxIters);

        for (int i = 0; i < maxIters; i++) {
            if (i > 0) {
                series = params.seriesStep(series, downgradeComplex(cur));
                cur = params.step(point, cur);
            }

            out[i].value = downgradeComplex(cur);
            for (int j = 0; j < 3; j++) {
                out[i].series[j] = series[j];
            }

            if (std::norm(cur) >= params.bailoutSqr()) {
                iters = min(iters, i+1);
            }
        }

        return iters;
    }

    template<typename Fractal>
    int computeMinIterations(const Fractal& params, DevComplex delta, const std::vector<RefPointInfo>& refData) {
        constexpr double MAX_ERROR = 0.002;
        int iters = 0;
        DevComplex cur = delta;

        while (iters < maxIters) {
            auto& ref = refData[iters];
            if ((cur+ref.value).norm() >= params.bailoutSqr()) {
                break;
            }

            DevComplex approx = ref.series.eval(delta);
            double error = 1 - max(abs(approx.x / cur.x), abs(approx.y / cur.y));
            if (isnan(error) || error > MAX_ERROR) {
                break;
            }

            cur = params.relativeStep(delta, cur, ref.value);
            iters++;
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
        info.scale = scale * 2 / width;

        std::vector<RefPointInfo> refData;
        info.approxIters = buildReferenceData(params, center, refData);
        devReferenceData.assign(refData);

        info.referenceData = devReferenceData.data();
        info.refPointScreen = DevComplex(width, height) * 0.5;

        if (useSeriesApproximation) {
            info.minIters = computeMinIterations(params, {-scale, 0}, refData);
            info.minIters = min(info.minIters, computeMinIterations(params, {scale, 0}, refData));
            info.minIters = min(info.minIters, computeMinIterations(params, {0, -scale}, refData));
            info.minIters = min(info.minIters, computeMinIterations(params, {0, scale}, refData));
            info.minIters = max(info.minIters-20, 0);
        } else {
            info.minIters = 0;
        }

        renderImageKernel<<<nBlocks, blockSize>>>(info);
        std::cout << "Skipped " << info.minIters << " iterations (useSeriesApproximation=" << useSeriesApproximation << ")" << std::endl;
    }

    double getScale() const {
        return scale;
    }

    void setScale(double val) {
        const unsigned digits10 = ceil(10 - log10(val));
        BigFloat::default_precision(digits10);
        
        // force update of center's precision
        center = BigComplex(
            BigFloat(center.real(), digits10),
            BigFloat(center.imag(), digits10));
        
        scale = val;
    }

private:
    CudaArray<RefPointInfo> devReferenceData;
    double scale;
public:
    Color *devImage;
    int maxIters, width, height;
    BigComplex center;
    bool useSeriesApproximation{true};
};
