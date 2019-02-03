#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

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

    void swap(ReferenceData& other) {
        std::swap(point, other.point);
        values.swap(other.values);
        series.swap(other.series);
        seriesErrors.swap(other.seriesErrors);
        devValues.swap(other.devValues);
    }
};

template<typename Fractal>
class Renderer : public BaseRenderer {
private:
    template<typename T>
    struct RendererImpl {
        RendererImpl(Renderer<Fractal>& parent) : view(parent), workerThread(&RendererImpl::referenceWorker, this) {}

        ~RendererImpl() {
            {
                std::unique_lock<std::mutex> lock(refMutex);
                workerShouldExit = true;
                refCond.notify_all();
            }
            workerThread.join();
        }

        void buildReferenceData(ReferenceData<T>& out) {
            BigComplex cur = out.point;
            out.values = { Complex<T>(cur) };
            out.series = { ExtComplex(1) };
            out.seriesErrors = { 0 };

            // Race condition on maxIters and params, theoretically
            for (int i = 1; i < view.maxIters && out.values.back().norm() < view.params.bailoutSqr(); i++) {
                out.series.push_back(view.params.seriesStep(out.series.back(), ExtComplex(cur)));
                cur = view.params.step(out.point, cur);
                out.values.push_back(Complex<T>(cur));

                ExtFloat error = 0;

                if (i >= SERIES_DEGREE) {
                    auto& curSeries = out.series.back();
                    error = curSeries[SERIES_DEGREE-1].norm() / curSeries[0].norm();
                }

                out.seriesErrors.push_back(std::max(out.seriesErrors.back(), error));
            }

            out.devValues.assign(out.values);
        }

        void referenceWorker() {
            ReferenceData<T> newRefData;

            while (true) {
                {
                    std::unique_lock<std::mutex> lock(refMutex);
                    refCond.wait(lock, [this] { return refDataRequest || workerShouldExit; });

                    if (workerShouldExit) {
                        return;
                    }

                    refDataRequest = false;
                    newRefData.point = newRefPoint;
                }

                buildReferenceData(newRefData);

                {
                    std::unique_lock<std::mutex> lock(refMutex);
                    refData.swap(newRefData);
                    refDataReady = true;
                    refCond.notify_all();
                }
            }
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

        void updateReference(std::unique_lock<std::mutex>& lock) {
            Complex<float> diff((refData.point - view.center) / view.scale);
            float dist = sqrt(diff.norm());

            if (dist > 0.05) {
                newRefPoint = view.center;
                refDataRequest = true;
                if (dist > 1000) {
                    refDataReady = false;
                }
                refCond.notify_one();
            }

            refCond.wait(lock, [this] { return refDataReady; });
        }

        void updateReference() {
            std::unique_lock<std::mutex> lock(refMutex);
            updateReference(lock);
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

            std::unique_lock<std::mutex> lock(refMutex);
            updateReference(lock);

            RenderInfo<Fractal, T> info;
            info.params = view.params;
            info.image = devImage;
            info.maxIters = view.maxIters;
            info.width = view.width;
            info.height = view.height;
            info.useSmoothing = view.useSmoothing;
            info.scale = fScale * 2 / view.width;

            info.deltaIters = std::min(view.maxIters, int(refData.devValues.size()));
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
        CudaVar<Series<ExtComplex>> devSeries;

        std::thread workerThread;
        std::mutex refMutex;
        std::condition_variable refCond;
        ReferenceData<T> refData;
        bool refDataReady{false}, refDataRequest{false}, workerShouldExit{false};
        BigComplex newRefPoint;
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
        if (scale > 1e-300) {
            implDouble.render(devImage);
        } else {
            implExtended.render(devImage);
        }
    }

    void preprocess() {
        implDouble.updateReference();
        implExtended.updateReference();
    }

private:
    RendererImpl<double> implDouble{*this};
    RendererImpl<ExtFloat> implExtended{*this};
public:
    Fractal params;
};
