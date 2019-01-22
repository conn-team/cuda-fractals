#pragma once

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
};
