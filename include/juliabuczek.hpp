#pragma once

#include <complex>
#include "series.hpp"

struct JuliaBuczek {
    JuliaBuczek(DevComplex seed = {-0.8, 0.156}) : seed(seed) {}

    __both__ double bailoutSqr() const {
        return 4;
    }

    template<typename T>
    T step(T, T previous) const {
        return previous*previous + T(seed.x, seed.y);
    }

    template<typename T>
    __both__ T relativeStep(T, T prevDelta, T prevReference) const {
        return prevDelta*(prevDelta + 2.0*prevReference);
    }

    template<typename T>
    CubicSeries<T> seriesStep(CubicSeries<T> prevSeries, T prevReference) const {
        return {
            T(2.0)*prevReference*prevSeries[0],
            T(2.0)*prevReference*prevSeries[1] + prevSeries[0]*prevSeries[0],
            T(2.0)*prevReference*prevSeries[2] + T(2.0)*prevSeries[0]*prevSeries[1]
        };
    }

    DevComplex seed;
};
