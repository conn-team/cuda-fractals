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
    virtual void resetScale() = 0;
    virtual void preprocess() = 0;

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
    struct RendererImpl {
        RendererImpl(Renderer<Fractal>& parent) : view(parent) {}

        void buildReferenceData(ReferenceData<T>& out, const BigComplex& point) {
            BigComplex cur = point;
            out.values = { Complex<T>(cur) };
            out.series = { ExtComplex(1) };
            out.seriesErrors = { 0 };

            for (int i = 1; i < view.maxIters && out.values.back().norm() < view.params.bailoutSqr(); i++) {
                out.series.push_back(view.params.seriesStep(out.series.back(), ExtComplex(cur)));
                cur = view.params.step(point, cur);
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

        int findMinIterations(ExtComplex delta) {
            constexpr double ESTIMATE_COEFF = 1;
            constexpr double MAX_ERROR = 0.002;

            // First find loose estimation

            ExtFloat invNorm = ExtFloat(1) / delta.norm();
            ExtFloat maxError = ESTIMATE_COEFF;

            for (int i = 1; i < SERIES_DEGREE; i++) {
                maxError *= invNorm;
            }

            auto found = std::lower_bound(refData.seriesErrors.begin(), refData.seriesErrors.end(), maxError);
            int iters = std::max(found - refData.seriesErrors.begin() - 100, 0L);

            // Now tighten bound

            ExtComplex cur = refData.series[iters].evaluate(delta);

            while (iters < int(refData.values.size())) {
                ExtComplex ref(refData.values[iters]);
                if (float((cur+ref).norm()) >= view.params.bailoutSqr()) {
                    break;
                }

                ExtComplex approx = refData.series[iters].evaluate(delta);
                double errorX = abs(1 - double(approx.x/cur.x));
                double errorY = abs(1 - double(approx.y/cur.y));

                if (std::isnan(errorX) || std::isnan(errorY) || max(errorX, errorY) > MAX_ERROR) {
                    break;
                }

                cur = view.params.relativeStep(delta, cur, ref);
                iters++;
            }

            return max(iters-10, 0);
        }

        void updateReference() {
            Complex<float> diff((refData.point - view.center) / view.scale);
            float dist = sqrt(diff.norm());

            if (dist > 0.5) {
                buildReferenceData(refData, view.center);
            }
        }

        void processStats(int skipped) {
            prefixAggregate<StatsEntry, StatsAggregate>(stats);
            StatsEntry entry = stats.get(stats.size()-1);

            view.skippedIters = skipped;
            view.realMinIters = entry.itersMin - skipped;
            view.realMaxIters = entry.itersMax - skipped;
            view.avgIters = double(entry.itersSum) / double(view.width*view.height) - skipped;
        }

        void render(Color *devImage) {
            constexpr uint32_t blockSize = 512;
            uint32_t nBlocks = (view.width*view.height+blockSize-1) / blockSize;
            T fScale(view.scale);

            updateReference();

            RenderInfo<Fractal, T> info;
            info.params = view.params;
            info.image = devImage;
            info.maxIters = view.maxIters;
            info.width = view.width;
            info.height = view.height;
            info.useSmoothing = view.useSmoothing;
            info.scale = fScale * 2 / view.width;

            info.approxIters = std::min(view.maxIters, int(refData.devValues.size()));
            info.referenceData = refData.devValues.data();
            info.refPointScreen = Complex<T>(view.pointToScreen(refData.point));

            if (view.useSeriesApproximation) {
                info.minIters = findMinIterations({fScale, fScale});
            } else {
                info.minIters = 0;
            }

            devSeries.set(refData.series[info.minIters]);
            info.series = devSeries.pointer();

            stats.resizeDiscard(view.width*view.height);
            info.stats = stats.data();

            renderImageKernel<<<nBlocks, blockSize>>>(info);
            processStats(info.minIters);
        }

        Renderer& view;
        CudaArray<StatsEntry> stats;
        ReferenceData<T> refData;
        CudaVar<Series<ExtComplex>> devSeries;
    };

public:
    Renderer(Fractal p = {}) : params(p) {
        reset();
    }

    void resetScale() {
        setScale(params.defaultScale());
    }

    void reset() {
        resetScale();
        center = BigComplex(params.defaultCenter());
        maxIters = params.defaultMaxIters();
    }

    void render(Color *devImage) {
        if (scale > 1e-30) {
            implFloat.render(devImage);
        } else if (scale > 1e-300) {
            implDouble.render(devImage);
        } else {
            implExtended.render(devImage);
        }
    }

    void preprocess() {
        implFloat.updateReference();
        implDouble.updateReference();
        implExtended.updateReference();
    }

private:
    RendererImpl<float> implFloat{*this};
    RendererImpl<double> implDouble{*this};
    RendererImpl<ExtFloat> implExtended{*this};
public:
    Fractal params;
};
