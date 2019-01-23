#pragma once

#include "series.hpp"

struct Mandelbrot {
    __both__ double bailoutSqr() const {
        return 4;
    }

    template<typename T>
    T step(T first, T previous) const {
        return previous*previous + first;
    }

    template<typename T>
    __both__ T relativeStep(T firstDelta, T prevDelta, T prevReference) const {
        return 2.0*prevDelta*prevReference + prevDelta*prevDelta + firstDelta;
    }

    template<typename T>
    CubicSeries<T> seriesStep(CubicSeries<T> prevSeries, T prevReference) const {
        return {
            2.0*prevReference*prevSeries[0] + 1.0,
            2.0*prevReference*prevSeries[1] + prevSeries[0]*prevSeries[0],
            2.0*(prevReference*prevSeries[2] + prevSeries[0]*prevSeries[1])
        };
    }
};
