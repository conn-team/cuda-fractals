#pragma once

#include <cstdint>

#include "cuda_helper.hpp"
#include "complex.hpp"
#include "color.hpp"
#include "series.hpp"

struct RefPointInfo {
    DevComplex value;
    CubicSeries<DevComplex> series;
};

template<typename Fractal>
class RenderInfo {
private:
    __device__ Color getColor(int iters, DevComplex end) {
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

        float subIters = iters % 160;
        if (useSmoothing) {
            subIters += 2 - log10f(end.norm()) / log10f(4.0f);
            if ((subIters = max(0.0f, subIters)) > 160.f) {
                subIters -= 160.f;
            }
        }
        subIters /= 32.f;

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
        DevComplex cur = referenceData[iters].series.evaluate(pos);

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

            cur = params.step(pos, cur);
            iters++;
        }

        image[index] = getColor(iters, cur);
    }

public:
    Fractal params;
    Color *image;
    RefPointInfo *referenceData;
    int minIters, maxIters, approxIters, width, height;
    DevComplex refPointScreen;
    bool useSmoothing;
    double scale;
};

template<typename Fractal>
__global__ void renderImageKernel(RenderInfo<Fractal> info) {
    info.render();
}
