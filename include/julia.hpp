#pragma once

#include "complex.hpp"
#include "series.hpp"

struct Julia {
    Julia(Complex<float> seed = {-0.8, 0.2}) : seed(seed) {}

    Complex<float> defaultCenter()   const { return {0, 0}; }
    float          defaultScale()    const { return 2; }
    int            defaultMaxIters() const { return 256; }
    __both__ float bailoutSqr()      const { return 4; }

    template<typename T>
    __both__ Complex<T> step(Complex<T>, Complex<T> previous) const {
        return previous.sqr() + Complex<T>(seed);
    }

    template<typename T>
    __both__ Complex<T> relativeStep(Complex<T>, Complex<T> prevDelta, Complex<T> prevReference) const {
        return prevDelta*(prevDelta + T(2.0)*prevReference);
    }

    Complex<float> seed;
};
