#pragma once

#include "cuda_helper.hpp"
#include "complex.hpp"

constexpr int SERIES_DEGREE = 16;
constexpr int SERIES_STEP = 100;

template<typename Fractal, typename T>
struct SeriesInfo {
    __device__ void computePoints() {
        int index = blockIdx.x*blockDim.x + threadIdx.x;
        if (index >= degree) {
            return;
        }

        double angle = 2*M_PI*index / degree;
        Complex<T> pos(cos(angle), sin(angle));
        pos *= scale;

        Complex<T> cur = pos;
        int iters = 0;
        outPoints[index] = cur;

        for (int i = 1; i < numSteps; i++) {
            for (int j = 0; j < SERIES_STEP; j++) {
                cur = params.relativeStep(pos, cur, referenceData[iters++]);
            }
            outPoints[i*degree + index] = cur;
        }
    }

    Fractal params;
    int numSteps, degree;
    T scale;
    Complex<T> *referenceData, *outPoints;
};

template<typename Fractal, typename T>
__global__ void computeSeriesPointsKernel(SeriesInfo<Fractal, T> info) {
    info.computePoints();
}

template<typename T>
__both__ T evaluatePolynomial(T *poly, int size, const T& point) {
    T ret(0);
    for (int i = size-1; i >= 0; i--) {
        ret = ret*point + poly[i];
    }
    return ret;
}
