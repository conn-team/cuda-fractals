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

    __both__ DevComplex mouseToCoords(int y, int x) const {
        x -= width / 2.0;
        y -= height / 2.0;
        return {
            double(center.x) + (2.0 * x / width) * scale,
            double(center.y) + (2.0 * y / width) * scale
        };
    }

    virtual void render(Color *devImage) = 0;
    virtual void reset() = 0;

protected:
    double scale;
public:
    BigComplex center;
    int width, height, maxIters, skippedIters;
    bool useSeriesApproximation{true};
    bool useSmoothing{false};
};

template<typename Fractal>
class Renderer : public BaseRenderer {
private:
    std::vector<DevComplex> buildReferenceData(const BigComplex& point) {
        BigComplex cur = point;
        std::vector<DevComplex> data = { DevComplex(cur) };

        for (int i = 1; i < maxIters && cur.norm() < params.bailoutSqr(); i++) {
            cur = params.step(point, cur);
            data.push_back(DevComplex(cur));
        }

        return data;
    }

    int computeMinIterations(DevComplex delta, const std::vector<DevComplex>& refData, CubicSeries<ExtComplex>& outSeries) {
        constexpr double MAX_ERROR = 0.002;

        int iters = 0;
        DevComplex cur = delta;
        std::vector<CubicSeries<ExtComplex>> series = { {ExtComplex(1), ExtComplex(0), ExtComplex(0)} };

        while (iters < int(refData.size())) {
            auto& ref = refData[iters];
            if ((cur+ref).norm() >= params.bailoutSqr()) {
                break;
            }

            DevComplex approx(series.back().evaluate(ExtComplex(delta)));
            double error = max(abs(1 - approx.x/cur.x), abs(1 - approx.y/cur.y));

            if (isnan(error) || error > MAX_ERROR) {
                break;
            }

            cur = params.relativeStep(delta, cur, ref);
            series.push_back(params.seriesStep(series.back(), ExtComplex(ref)));
            iters++;
        }

        iters = max(0, iters-10);
        outSeries = series[iters];
        return int(iters);
    }

public:
    Renderer(Fractal p = {}) : params(p) {
        reset();
    }

    void reset() {
        setScale(params.defaultScale());
        center = BigComplex(params.defaultCenter());
        maxIters = params.defaultMaxIters();
    }

    void render(Color *devImage) {
        constexpr uint32_t blockSize = 512;
        uint32_t nBlocks = (width*height+blockSize-1) / blockSize;

        RenderInfo<Fractal> info;
        info.params = params;
        info.image = devImage;
        info.maxIters = maxIters;
        info.width = width;
        info.height = height;
        info.useSmoothing = useSmoothing;
        info.scale = scale * 2 / width;

        std::vector<DevComplex> refData = buildReferenceData(center);
        devReferenceData.assign(refData);

        info.approxIters = int(refData.size());
        info.referenceData = devReferenceData.data();
        info.refPointScreen = DevComplex(width, height) * 0.5;

        if (useSeriesApproximation) {
            info.minIters = computeMinIterations({scale, scale}, refData, info.series);
        } else {
            info.minIters = 0;
            info.series = { ExtComplex(1), ExtComplex(0), ExtComplex(0) };
        }

        skippedIters = info.minIters;
        renderImageKernel<<<nBlocks, blockSize>>>(info);
    }

private:
    CudaArray<DevComplex> devReferenceData;
public:
    Fractal params;
};
