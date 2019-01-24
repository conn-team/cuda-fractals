#pragma once

#include <complex>
#include "series.hpp"


    template<typename T>
    virtual __both__ T step(T, T previous) const = 0;

    template<typename T>
    virtual __both__ T relativeStep(T, T prevDelta, T prevReference) const = 0;

    template<typename T>
    virtual CubicSeries<T> seriesStep(CubicSeries<T> prevSeries, T prevReference) = 0;

    DevComplex seed;
};
