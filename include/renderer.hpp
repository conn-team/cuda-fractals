#pragma once

#include <algorithm>
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

    virtual void render(Color *devImage) = 0;
    virtual void reset() = 0;

    Complex<float> pointToScreen(const BigComplex& p) {
        Complex<float> scaled((p-center) / scale);
        scaled *= width / 2.f;
        return { scaled.x + width / 2.f, scaled.y + height / 2.f };
    }

protected:
    BigFloat scale;
public:
    BigComplex center;
    int width, height, maxIters;
    bool useSeriesApproximation{true};
    bool useSmoothing{false};

    // Statistics
    int skippedIters{0}, realMinIters{0}, realMaxIters{0};
    double avgIters{0};
};

template<typename T>
struct ReferenceData {
    BigComplex point{1e9, 1e9};             // Reference point
    std::vector<Complex<T>> values;         // Reference point iterations
    std::vector<Series<ExtComplex>> series; // Series approximation for delta iterations
    std::vector<ExtFloat> seriesErrors;     // Binary-searchable series error estimates
    CudaArray<Complex<T>> devValues;        // Device array for iterations
};

template<typename Fractal>
class Renderer : public BaseRenderer {
private:
    template<typename T>
    void buildReferenceData(ReferenceData<T>& out, const BigComplex& point) {
        BigComplex cur = point;
        out.values = { Complex<T>(cur) };
        out.series = { ExtComplex(1) };
        out.seriesErrors = { 0 };

        for (int i = 1; i < maxIters && cur.norm() < params.bailoutSqr(); i++) {
            out.series.push_back(params.seriesStep(out.series.back(), ExtComplex(cur)));
            cur = params.step(point, cur);
            out.values.push_back(Complex<T>(cur));

            ExtFloat error = 0;

            if (i >= SERIES_DEGREE) {
                auto& curSeries = out.series.back();
                error = curSeries[SERIES_DEGREE-1].norm() / curSeries[0].norm();
            }

            out.seriesErrors.push_back(std::max(out.seriesErrors.back(), error));
        }

        out.devValues.assign(out.values);
        out.point = point;
    }

    template<typename T>
    int findMinIterations(ExtComplex delta, const ReferenceData<T>& refData) {
        constexpr double ESTIMATE_COEFF = 1;
        constexpr double MAX_ERROR = 0.002;

        // First find loose estimation

        ExtFloat invNorm = ExtFloat(1) / delta.norm();
        ExtFloat maxError = ESTIMATE_COEFF;

        for (int i = 1; i < SERIES_DEGREE; i++) {
            maxError *= invNorm;
        }

        auto found = std::lower_bound(refData.seriesErrors.begin(), refData.seriesErrors.end(), maxError);
        int iters = std::max(found - refData.seriesErrors.begin() - 10, 0L);

        // Now tighten bound

        ExtComplex cur = refData.series[iters].evaluate(delta);

        while (iters < int(refData.values.size())) {
            ExtComplex ref(refData.values[iters]);
            if (float((cur+ref).norm()) >= params.bailoutSqr()) {
                break;
            }

            ExtComplex approx = refData.series[iters].evaluate(delta);
            double errorX = abs(1 - double(approx.x/cur.x));
            double errorY = abs(1 - double(approx.y/cur.y));

            if (std::isnan(errorX) || std::isnan(errorY) || max(errorX, errorY) > MAX_ERROR) {
                break;
            }

            cur = params.relativeStep(delta, cur, ref);
            iters++;
        }

        return max(iters-10, 0);
    }

    template<typename T>
    void updateReference(ReferenceData<T>& refData) {
        Complex<float> diff((refData.point - center) / scale);
        float dist = sqrt(diff.norm());

        if (dist > 0.5) {
            buildReferenceData(refData, center);
        }
    }

    void processStats(int skipped) {
        prefixAggregate<StatsEntry, StatsAggregate>(stats);
        StatsEntry entry = stats.get(stats.size()-1);

        skippedIters = skipped;
        realMinIters = entry.itersMin - skipped;
        realMaxIters = entry.itersMax - skipped;
        avgIters = double(entry.itersSum) / double(width*height) - skipped;
    }

    template<typename T>
    void performRender(Color *devImage, ReferenceData<T>& refData) {
        constexpr uint32_t blockSize = 512;
        uint32_t nBlocks = (width*height+blockSize-1) / blockSize;
        T fScale(scale);

        updateReference(refData);

        RenderInfo<Fractal, T> info;
        info.params = params;
        info.image = devImage;
        info.maxIters = maxIters;
        info.width = width;
        info.height = height;
        info.useSmoothing = useSmoothing;
        info.scale = fScale * 2 / width;

        info.approxIters = int(refData.devValues.size());
        info.referenceData = refData.devValues.data();
        info.refPointScreen = Complex<T>(pointToScreen(refData.point));

        if (useSeriesApproximation) {
            info.minIters = findMinIterations({fScale, fScale}, refData);
        } else {
            info.minIters = 0;
        }

        info.series = refData.series[info.minIters];

        stats.resizeDiscard(width*height);
        info.stats = stats.data();

        renderImageKernel<<<nBlocks, blockSize>>>(info);
        processStats(info.minIters);
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
        if (scale > 1e-300) {
            performRender(devImage, refDataDouble);
        } else {
            performRender(devImage, refDataExtended);
        }
    }

private:
    CudaArray<StatsEntry> stats;
    ReferenceData<double> refDataDouble;
    ReferenceData<ExtFloat> refDataExtended;
public:
    Fractal params;
};
