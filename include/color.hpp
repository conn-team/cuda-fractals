#pragma once

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

struct HSVAColor {
    union {
        uint32_t value;
        struct {
            float h, s, v, a;
        };
    };

    __both__ HSVAColor(float h, float s, float v, float a = 1.0) : h(h), s(s), v(v), a(a) {}
    __both__ HSVAColor(void) : HSVAColor(0.0, 0.0, 0.0) {}

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
