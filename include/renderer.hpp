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

    template<typename T>
    void performRender(Color *devImage, CudaArray<Complex<T>>& devRefData) {
        constexpr uint32_t blockSize = 512;
        uint32_t nBlocks = (width*height+blockSize-1) / blockSize;

        RenderInfo<Fractal, T> info;
        info.params = params;
        info.image = devImage;
        info.maxIters = maxIters;
        info.width = width;
        info.height = height;
        info.useSmoothing = useSmoothing;
        info.scale = scale * 2 / width;

        std::vector<Complex<T>> refData = buildReferenceData<T>(center);
        devRefData.assign(refData);

        info.approxIters = int(refData.size());
        info.referenceData = devRefData.data();
        info.refPointScreen = Complex<T>(width, height) * 0.5;

        if (useSeriesApproximation) {
            info.minIters = computeMinIterations({scale, scale}, refData, info.series);
        } else {
            info.minIters = 0;
            info.series = { ExtComplex(1), ExtComplex(0), ExtComplex(0) };
        }

        skippedIters = info.minIters;
        renderImageKernel<<<nBlocks, blockSize>>>(info);
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
        } else {
            performRender(devImage, refDataDouble);
        }
    }

private:
    CudaArray<Complex<float>> refDataFloat;
    CudaArray<Complex<double>> refDataDouble;
    CudaArray<Complex<ExtFloat>> refDataExtended;
public:
    Fractal params;
};
