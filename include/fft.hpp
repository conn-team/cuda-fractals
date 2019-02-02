#pragma once

#include <vector>
#include "complex.hpp"

template<typename T>
class FFT {
public:
    void inverse(std::vector<Complex<T>>& buf) {
        if (n != int(buf.size())) {
            init(buf.size());
        }

        for (int i = 0; i < bits; i++) {
            int shift = 1 << (bits-i-1);

            for (int j = 0; j < (1<<i); j++) {
                for (int k = 0; k < shift; k++) {
                    int a = (j << (bits-i)) | k;
                    int b = a | shift;
                    Complex<T> v1 = buf[a], v2 = buf[b];
                    Complex<T> base = bases[(k<<i) & (n-1)];

                    buf[b] = (v1 - v2) * base;
                    buf[a] = v1 + v2;
                }
            }
        }

        for (uint32_t i = 0; i < buf.size(); i++) {
            uint32_t j = reverseBits(i);
            if (i < j) {
                std::swap(buf[i], buf[j]);
            }
        }

        for (auto& x : buf) {
            x /= T(n);
        }
    }

private:
    uint32_t reverseBits(uint32_t k) {
        k = (((k & 0xaaaaaaaa) >> 1) | ((k & 0x55555555) << 1));
        k = (((k & 0xcccccccc) >> 2) | ((k & 0x33333333) << 2));
        k = (((k & 0xf0f0f0f0) >> 4) | ((k & 0x0f0f0f0f) << 4));
        k = (((k & 0xff00ff00) >> 8) | ((k & 0x00ff00ff) << 8));
        return ((k >> 16) | (k << 16)) >> (32 - bits);
    }

    void init(int len) {
        assert(__builtin_popcount(len) == 1); // Must be power of 2
        n = len;
        bits = 31-__builtin_clz(len);

        double ang = -2*M_PI / len;
        Complex<T> base(cos(ang), sin(ang));
        bases.resize(len);
        bases[0] = Complex<T>(1);

        for (int i = 1; i < len; i++) {
            bases[i] = bases[i-1] * base;
        }
    }

    std::vector<Complex<T>> bases;
    int n, bits;
};
