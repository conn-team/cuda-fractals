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
#include "fft.hpp"

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
    BigComplex point{1e9, 1e9};                  // Reference point
    BigFloat scale{100};                         // Reference scale
    std::vector<Complex<T>> values;              // Reference point iterations
    std::vector<std::vector<ExtComplex>> series; // Series approximation for delta iterations (one per SERIES_STEP iterations)
    CudaArray<Complex<T>> devValues;             // Device array for reference point iterations
};

template<typename Fractal>
class Renderer : public BaseRenderer {
private:
    template<typename T>
    struct RendererImpl {
        RendererImpl(Renderer<Fractal>& parent) : view(parent) {}

        void computeReferenceValues() {
            BigComplex cur = view.center;
            refData.point = view.center;
            refData.values.clear();
            refData.values.push_back(Complex<T>(cur));

            for (int i = 1; i < view.maxIters && cur.norm() < view.params.bailoutSqr(); i++) {
                cur = view.params.step(view.center, cur);
                refData.values.push_back(Complex<T>(cur));
            }

            refData.devValues.assign(refData.values);
        }

        void computeSeries() {
            constexpr double RADIUS = 0.5;

            // Compute points around reference
            int iters = refData.values.size();
            refData.scale = view.scale;

            SeriesInfo<Fractal, T> info;
            info.params = view.params;
            info.numSteps = (iters+SERIES_STEP-1) / SERIES_STEP;
            info.degree = SERIES_DEGREE;
            info.scale = T(view.scale) * RADIUS;
            info.referenceData = refData.devValues.data();

            CudaArray<Complex<T>> devPoints(info.numSteps * info.degree);
            info.outPoints = devPoints.data();

            computeSeriesPointsKernel<<<1, info.degree>>>(info);

            std::vector<Complex<T>> points;
            devPoints.get(points);

            // Interpolate points using FFT
            ExtFloat invScale(1.0 / RADIUS / view.scale);
            FFT<ExtFloat> fft;
            refData.series.resize(info.numSteps);

            for (int i = 0; i < info.numSteps; i++) {
                auto& series = refData.series[i];
                series.resize(info.degree);

                for (int j = 0; j < info.degree; j++) {
                    series[j] = ExtComplex(points[i*info.degree + j]);
                }

                ExtFloat mult(1.0);
                fft.inverse(series);

                for (auto& x : series) {
                    x *= mult;
                    mult *= invScale;
                }
            }
        }

        int findMaxSkip(ExtComplex delta) {
            constexpr double MAX_ERROR = 0.002;

            ExtComplex cur = delta;
            int iters = 0;

            for (size_t i = 1; i < refData.series.size(); i++) {
                for (int j = 0; j < SERIES_STEP; j++) {
                    cur = view.params.relativeStep(delta, cur, ExtComplex(refData.values[iters++]));
                }

                auto& series = refData.series[i];
                ExtComplex approx = evaluatePolynomial(series.data(), series.size(), delta);
                double errorX = abs(1 - double(approx.x/cur.x));
                double errorY = abs(1 - double(approx.y/cur.y));

                if (std::max(errorX, errorY) > MAX_ERROR) {
                    return i-1;
                }
            }

            return refData.series.size()-1;
        }

        void updateReference() {
            Complex<float> diff((refData.point - view.center) / view.scale);
            float dist = sqrt(diff.norm());

            if (dist > 0.5) {
                computeReferenceValues();
            }
            computeSeries();
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
                int skip = findMaxSkip({fScale, fScale});
                devSeries.assign(refData.series[skip]);
                info.minIters = skip * SERIES_STEP;
                info.seriesDegree = SERIES_DEGREE;
            } else {
                devSeries.assign(refData.series[0]);
                info.minIters = 0;
                info.seriesDegree = 2;
            }

            stats.resizeDiscard(view.width*view.height);
            info.stats = stats.data();
            info.series = devSeries.data();

            renderImageKernel<<<nBlocks, blockSize>>>(info);
            processStats(info.minIters);
        }

        Renderer& view;
        ReferenceData<T> refData;
        CudaArray<StatsEntry> stats;
        CudaArray<ExtComplex> devSeries;
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
