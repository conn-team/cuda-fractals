#pragma once

#include <cstdint>
#include <iostream>

#include "cuda_helper.hpp"
#include "complex.hpp"
#include "color.hpp"
#include "series.hpp"
#include "bignum.hpp"
#include "renderer_dev.hpp"

class BaseRenderer {
public:
    double getScale() const {
        return scale;
    }

    void setScale(double val) {
        // Set scale and adapt BigFloat precision
        const long digits10 = lround(10 - log10(val));
        BigFloat::default_precision(digits10);
        center = { BigFloat(center.x, digits10), BigFloat(center.y, digits10) };
        scale = val;
    }

    virtual void render(Color *devImage) = 0;

protected:
    double scale;
public:
    BigComplex center;
    int width, height, maxIters;
    bool useSeriesApproximation{true};
    bool useSmoothing{false};
};

template<typename Fractal>
class Renderer : public BaseRenderer {
private:
    std::vector<RefPointInfo> buildReferenceData(const BigComplex& point) {
        BigComplex cur = point;
        CubicSeries<DevComplex> series{1, 0, 0};
        std::vector<RefPointInfo> data;

        for (int i = 0; i < maxIters; i++) {
            if (i > 0) {
                if (useSeriesApproximation) {
                    series = params.seriesStep(series, data[i-1].value);
                }
                cur = params.step(point, cur);
            }

            data.push_back({ DevComplex(cur), series });

            if (cur.norm() >= params.bailoutSqr()) {
                break;
            }
        }

        return data;
    }

    int computeMinIterations(DevComplex delta, const std::vector<RefPointInfo>& refData) {
        constexpr double MAX_ERROR = 0.002;
        size_t iters = 0;
        DevComplex cur = delta;

        while (iters < refData.size()) {
            auto& ref = refData[iters];
            if ((cur+ref.value).norm() >= params.bailoutSqr()) {
                break;
            }

            DevComplex approx = ref.series.evaluate(delta);
            double error = max(abs(1 - approx.x/cur.x), abs(1 - approx.y/cur.y));

            if (isnan(error) || error > MAX_ERROR) {
                break;
            }

            cur = params.relativeStep(delta, cur, ref.value);
            iters++;
        }

        return int(iters);
    }

public:
    void render(Color *devImage) {
        constexpr uint32_t blockSize = 1024;
        uint32_t nBlocks = (width*height+blockSize-1) / blockSize;

        RenderInfo<Fractal> info;
        info.params = params;
        info.image = devImage;
        info.maxIters = maxIters;
        info.width = width;
        info.height = height;
        info.useSmoothing = useSmoothing;
        info.scale = scale * 2 / width;

        std::vector<RefPointInfo> refData = buildReferenceData(center);
        devReferenceData.assign(refData);

        info.approxIters = int(refData.size());
        info.referenceData = devReferenceData.data();
        info.refPointScreen = DevComplex(width, height) * 0.5;

        if (useSeriesApproximation) {
            info.minIters = computeMinIterations({-scale, 0}, refData);
            info.minIters = min(info.minIters, computeMinIterations({scale, 0}, refData));
            info.minIters = min(info.minIters, computeMinIterations({0, -scale}, refData));
            info.minIters = min(info.minIters, computeMinIterations({0, scale}, refData));
            info.minIters = max(info.minIters-20, 0);
        } else {
            info.minIters = 0;
        }

        renderImageKernel<<<nBlocks, blockSize>>>(info);
        std::cout << "Skipped " << info.minIters << " iterations (useSeriesApproximation=" << useSeriesApproximation << ")" << std::endl;
    }

private:
    CudaArray<RefPointInfo> devReferenceData;
public:
    Fractal params;
};
