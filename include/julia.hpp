#pragma once

#include "complex.hpp"
#include "series.hpp"

struct Julia {
    Julia(DevComplex seed = {-0.8, 0.2}) : seed(seed) {}

    BigComplex      defaultCenter()   const { return {-0.7, 0}; }
    double          defaultScale()    const { return 1.5; }
    int             defaultMaxIters() const { return 250; }
    __both__ double bailoutSqr()      const { return 4; }

    #pragma hd_warning_disable
    template<typename T>
    __both__ Complex<T> step(Complex<T>, Complex<T> previous) const {
        return previous.sqr() + Complex<T>(seed);
    }

    template<typename T>
    __both__ Complex<T> relativeStep(Complex<T>, Complex<T> prevDelta, Complex<T> prevReference) const {
        return prevDelta*(prevDelta + 2.0*prevReference);
    }

    template<typename T>
    CubicSeries<Complex<T>> seriesStep(CubicSeries<Complex<T>> prevSeries, Complex<T> prevReference) const {
        return {
            2.0*prevReference*prevSeries[0],
            2.0*prevReference*prevSeries[1] + prevSeries[0].sqr(),
            2.0*prevReference*prevSeries[2] + 2.0*prevSeries[0]*prevSeries[1]
        };
    }

    DevComplex seed;
};