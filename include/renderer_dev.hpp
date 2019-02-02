#pragma once

#include <cstdint>

#include "cuda_helper.hpp"
#include "complex.hpp"
#include "color.hpp"
#include "series.hpp"

struct StatsEntry {
    int64_t itersSum;
    int itersMin, itersMax;
};

struct StatsAggregate {
    __device__ StatsEntry operator()(const StatsEntry& l, const StatsEntry& r) const {
        return { l.itersSum+r.itersSum, min(l.itersMin, r.itersMin), max(l.itersMax, r.itersMax) };
    }
};

template<typename Fractal, typename T>
class RenderInfo {
public:
    __both__ Complex<T> screenToDelta(int x, int y) {
        return (Complex<T>(x, y) - refPointScreen) * scale;
    }

    __device__ void render() {
        int index = blockIdx.x*blockDim.x + threadIdx.x;
        int x = index % width, y = index / width;
        if (y >= height) {
            return;
        }

        Complex<T> pos = screenToDelta(x, y);
        Complex<T> cur(series->evaluate(ExtComplex(pos)));
        int iters = minIters;

        // First, try to make multiple steps to avoid checking bailout often
        constexpr int STEP = 5;

        while (iters+STEP < deltaIters) {
            Complex<T> tmp = cur;

            #pragma unroll
            for (int i = 0; i < STEP; i++) {
                tmp = params.relativeStep(pos, tmp, referenceData[iters++]);
            }

            if (float((tmp+referenceData[iters]).norm()) >= params.bailoutSqr()) {
                iters -= STEP;
                break;
            }
            cur = tmp;
        }

        // Now step one by one
        while (iters+1 < deltaIters) {
            auto ref = referenceData[iters];
            if (float((cur+ref).norm()) >= params.bailoutSqr()) {
                break;
            }

            cur = params.relativeStep(pos, cur, ref);
            iters++;
        }

        // Reference data ended, switch to absolute calculations
        pos += referenceData[0];
        cur += referenceData[iters];

        while (iters < maxIters) {
            if (float(cur.norm()) >= params.bailoutSqr()) {
                break;
            }

            cur = params.step(pos, cur);
            iters++;
        }

        image[index] = Color::fromIterations(iters, Complex<float>(cur), maxIters, useSmoothing);
        stats[index] = { iters, iters, iters };
    }

public:
    Fractal params;
    Color *image;
    int minIters, maxIters, deltaIters, width, height;
    Complex<T> refPointScreen;
    bool useSmoothing;
    T scale;

    Complex<T> *referenceData;
    Series<ExtComplex> *series;
    StatsEntry *stats;
};

template<typename Fractal, typename T>
__global__ void renderImageKernel(RenderInfo<Fractal, T> info) {
    info.render();
}
