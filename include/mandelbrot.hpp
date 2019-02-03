#pragma once

#include "complex.hpp"
#include "series.hpp"

struct Mandelbrot {
    Complex<float> defaultCenter()   const { return {-0.7, 0}; }
    float          defaultScale()    const { return 1.5; }
    int            defaultMaxIters() const { return 256; }
    __both__ float bailoutSqr()      const { return 4; }

    template<typename T>
    __both__ Complex<T> step(const Complex<T>& first, const Complex<T>& previous) const {
        return previous.sqr() + first;
    }

    template<typename T>
    __both__ Complex<T> relativeStep(const Complex<T>& firstDelta, const Complex<T>& prevDelta, const Complex<T>& prevReference) const {
        return prevDelta*(prevDelta + T(2.0)*prevReference) + firstDelta;
    }

    template<typename T>
    Series<Complex<T>> seriesStep(const Series<Complex<T>>& prevSeries, const Complex<T>& prevReference) const {
        Series<Complex<T>> ret;

        #pragma omp parallel for
        for (int i = 0; i < SERIES_DEGREE; i++) {
            ret[i] = prevSeries[i] * prevReference;

            for (int j = 0; j < i/2; j++) {
                ret[i] += prevSeries[j] * prevSeries[i-j-1];
            }

            ret[i] *= T(2);

            if (i % 2) {
                ret[i] += prevSeries[i/2].sqr();
            } else if (i == 0) {
                ret[i] += T(1);
            }
        }

        return ret;
    }
};
