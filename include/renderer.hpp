#pragma once

#include <cstdint>
#include <iostream>

#include "cuda_helper.hpp"
#include "complex.hpp"
#include "color.hpp"
#include "series.hpp"
#include "bignum.hpp"
#include "renderer_dev.hpp"
#include "prefix_engine.hpp"

class BaseRenderer {
public:
    BigFloat getScale() const {
        return scale;
    }

    void setScale(const BigFloat& val) {
        // Set scale and adapt BigFloat precision
        const long digits10 = max(30L, lround(10 - log10(val)));
        BigFloat::default_precision(digits10);
        center = { BigFloat(center.x, digits10), BigFloat(center.y, digits10) };
        scale = val;
    }

    __both__ Complex<float> mouseToCoords(int y, int x) const {
        x -= width / 2.f;
        y -= height / 2.f;
        return {
            float(center.x) + (2.f * x / width) * float(scale),
            float(center.y) + (2.f * y / width) * float(scale)
        };
    }

    virtual void render(Color *devImage) = 0;
    virtual void reset() = 0;

protected:
    BigFloat scale;
public:
    BigComplex center;
    int width, height, maxIters;
    bool useSeriesApproximation{true};
    bool useSmoothing{false};

    int skippedIters{0}, realMaxIters{0};
    double avgIters{0};
};

template<typename Fractal>
class Renderer : public BaseRenderer {
private:
    template<typename T>
    std::vector<Complex<T>> buildReferenceData(const BigComplex& point) {
        BigComplex cur = point;
        std::vector<Complex<T>> data = { Complex<T>(cur) };

        for (int i = 1; i < maxIters && cur.norm() < params.bailoutSqr(); i++) {
            cur = params.step(point, cur);
            data.push_back(Complex<T>(cur));
        }

        return data;
    }

    template<typename T>
    int computeMinIterations(ExtComplex delta, const std::vector<Complex<T>>& refData, CubicSeries<ExtComplex>& outSeries) {
        constexpr double MAX_ERROR = 0.002;

        int iters = 0;
        ExtComplex cur = delta;
        std::vector<CubicSeries<ExtComplex>> series = { {ExtComplex(1), ExtComplex(0), ExtComplex(0)} };

        while (iters < int(refData.size())) {
            ExtComplex ref(refData[iters]);
            if (float((cur+ref).norm()) >= params.bailoutSqr()) {
                break;
            }

            ExtComplex approx = series.back().evaluate(delta);
            double errorX = abs(1 - double(approx.x/cur.x));
            double errorY = abs(1 - double(approx.y/cur.y));

            if (std::isnan(errorX) || std::isnan(errorY) || max(errorX, errorY) > MAX_ERROR) {
                break;
            }

            cur = params.relativeStep(delta, cur, ref);
            series.push_back(params.seriesStep(series.back(), ref));
            iters++;
        }

        iters = max(0, iters-10);
        outSeries = series[iters];
        return int(iters);
    }

    StatsEntry aggregateStats() {
        prefixAggregate<StatsEntry, StatsAggregate>(stats);
        return stats.get(stats.size()-1);
    }

    template<typename T>
    void performRender(Color *devImage, CudaArray<Complex<T>>& devRefData) {
        constexpr uint32_t blockSize = 512;
        uint32_t nBlocks = (width*height+blockSize-1) / blockSize;

        T fScale(scale);

        RenderInfo<Fractal, T> info;
        info.params = params;
        info.image = devImage;
        info.maxIters = maxIters;
        info.width = width;
        info.height = height;
        info.useSmoothing = useSmoothing;
        info.scale = fScale * 2 / width;

        std::vector<Complex<T>> refData = buildReferenceData<T>(center);
        devRefData.assign(refData);

        info.approxIters = int(refData.size());
        info.referenceData = devRefData.data();
        info.refPointScreen = Complex<T>(width, height) * 0.5;

        if (useSeriesApproximation) {
            info.minIters = computeMinIterations({fScale, fScale}, refData, info.series);
        } else {
            info.minIters = 0;
            info.series = { ExtComplex(1), ExtComplex(0), ExtComplex(0) };
        }

        skippedIters = info.minIters;
        stats.resizeDiscard(width*height);
        info.stats = stats.data();

        renderImageKernel<<<nBlocks, blockSize>>>(info);

        StatsEntry entry = aggregateStats();
        realMaxIters = entry.itersMax;
        avgIters = double(entry.itersSum) / double(width*height);
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
        if (scale > 1e-32) {
            performRender(devImage, refDataFloat);
        } else if (scale > 1e-300) {
            performRender(devImage, refDataDouble);
        } else {
            performRender(devImage, refDataExtended);
        }
    }

private:
    CudaArray<StatsEntry> stats;
    CudaArray<Complex<float>> refDataFloat;
    CudaArray<Complex<double>> refDataDouble;
    CudaArray<Complex<ExtFloat>> refDataExtended;
public:
    Fractal params;
};
