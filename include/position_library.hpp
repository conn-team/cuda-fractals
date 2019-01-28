#pragma once

#include <vector>

namespace position_library {
    struct Position {
        int maxIters;
        const char *scale, *real, *imag;
    };

    extern std::vector<Position> mandelbrot;
};
