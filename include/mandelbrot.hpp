#pragma once

#include "complex.hpp"
#include "series.hpp"

struct Mandelbrot {
    Complex<float> defaultCenter()   const { return {-0.7, 0}; }
    float          defaultScale()    const { return 1.5; }
    int            defaultMaxIters() const { return 256; }
    __both__ float bailoutSqr()      const { return 4; }

    template<typename T>
    __both__ Complex<T> step(Complex<T> first, Complex<T> previous) const {
        return previous.sqr() + first;
    }

    template<typename T>
    __both__ Complex<T> relativeStep(Complex<T> firstDelta, Complex<T> prevDelta, Complex<T> prevReference) const {
        return prevDelta*(prevDelta + T(2.0)*prevReference) + firstDelta;
    }
};
