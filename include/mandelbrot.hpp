#pragma once

struct Mandelbrot {
    double bailout() {
        return 2;
    }

    template<typename T>
    T step(T previous, T pos) {
        return previous.sqr() + pos;
    }

    template<typename T>
    T relativeStep(T prevDelta, T prevReference, T relPos) {
        return 2*prevDelta*prevReference + prevDelta*prevDelta + relPos;
    }
};
