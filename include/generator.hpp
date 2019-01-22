#pragma once

#include <complex>
#include <cstdint>
#include <gmpxx.h>

#include "cuda_helper.hpp"
#include "complex.hpp"

struct Viewport {
    Color *image;
    int width, height;
    std::complex<mpf_class> center;
    mpf_class scale;
};

void renderImage(const Viewport& view);
