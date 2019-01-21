#pragma once

#include <cstdint>

#include "cuda_helper.hpp"
#include "complex.hpp"

struct Color {
    union {
        uint32_t value;
        struct {
            uint8_t b, g, r, a;
        };
    };

    __both__ Color(uint32_t value = 0xFF000000) : value(value) {}
    __both__ Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xFF) : r(r), g(g), b(b), a(a) {}
};

struct Viewport {
    Color *image;
    int width, height;
    Complex<double> center;
    double scale;
};

void renderImage(const Viewport& view);
