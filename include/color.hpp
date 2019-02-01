#pragma once

#include "cuda_helper.hpp"
#include <cstdint>

struct Color {
    union {
        uint32_t value;
        struct {
            uint8_t b, g, r, a;
        };
    };

    __both__ constexpr Color(uint32_t value = 0xFF000000) : value(value) {}
    __both__ constexpr Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xFF) : r(r), g(g), b(b), a(a) {}

    __both__ static Color fromIterations(int iters, Complex<float> end, int maxIters, bool smooth) {
        if (iters == maxIters) {
            return Color(0, 0, 0);
        }

        Color colors[] = {
            {0, 0, 0},
            {200, 0, 0},
            {200, 200, 0},
            {0, 200, 200},
            {0, 0, 200},
        };

        float subIters = iters % 160;
        if (smooth) {
            subIters += 2 - log10f(end.norm()) / log10f(4.0f);
            if ((subIters = max(0.0f, subIters)) > 160.f) {
                subIters -= 160.f;
            }
        }
        subIters /= 32.f;

        int c0 = floor(subIters - 5.0f * floor(subIters / 5.0f));
        int c1 = ceil (subIters - 5.0f * floor(subIters / 5.0f));
        float t = subIters - c0;
        c1 %= 5;

        return {
            static_cast<uint8_t>((1.0f - t) * colors[c0].r + t * colors[c1].r),
            static_cast<uint8_t>((1.0f - t) * colors[c0].g + t * colors[c1].g),
            static_cast<uint8_t>((1.0f - t) * colors[c0].b + t * colors[c1].b)
        };
    }
};

struct HSVAColor {
    float h, s, v, a;

    __both__ constexpr HSVAColor(float h, float s, float v, float a = 1.0) : h(h), s(s), v(v), a(a) {}
    __both__ constexpr HSVAColor(void) : HSVAColor(0.0, 0.0, 0.0) {}

    __both__ Color toRGBA() const {
        if (h < 0 || h > 360) {
            return Color(0, 0, 0);
        }

        float c = v * s;
        float h2 = h / 60.0;
        int8_t sector = static_cast<int8_t>(h2);

        float x = h2 - 2 * floor(h2 / 2) - 1;
        x = c * (1 - (x < 0 ? -x : x));
        
        Color rgba;

        if        (sector < 1)   { rgba = Color(v           * 255.0, (x + v - c) * 255.0, (v - c)     * 255.0, a * 255.0); }
        else if   (sector < 2)   { rgba = Color((x + v - c) * 255.0, v           * 255.0, (v - c)     * 255.0, a * 255.0); }
        else if   (sector < 3)   { rgba = Color((v - c)     * 255.0, v           * 255.0, (x + v - c) * 255.0, a * 255.0); }
        else if   (sector < 4)   { rgba = Color((v - c)     * 255.0, (x + v - c) * 255.0, v           * 255.0, a * 255.0); }
        else if   (sector < 5)   { rgba = Color((x + v - c) * 255.0, (v - c)     * 255.0, v           * 255.0, a * 255.0); }
        else /*if (sector < 6)*/ { rgba = Color(v           * 255.0, (v - c)     * 255.0, (x + v - c) * 255.0, a * 255.0); }

        return rgba;
    }
};
