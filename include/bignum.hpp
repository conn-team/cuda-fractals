#pragma once

#ifdef __CUDA_ARCH__

// Dummy class to eliminate warnings about usage of BigFloat when using __both__
struct BigFloat {
    __both__ BigFloat() {}
    template<typename T> __both__ BigFloat(const T&) {}
    template<typename T> __both__ BigFloat(const T&, int prec) {}

    __both__ operator double() const { return 0; }

    template<typename T> __both__ BigFloat operator+(const T&) const { return {}; }
    template<typename T> __both__ BigFloat operator-(const T&) const { return {}; }
    template<typename T> __both__ BigFloat operator*(const T&) const { return {}; }
    template<typename T> __both__ BigFloat operator/(const T&) const { return {}; }

    template<typename T> __both__ BigFloat& operator+=(const T&) { return *this; }
    template<typename T> __both__ BigFloat& operator-=(const T&) { return *this; }
    template<typename T> __both__ BigFloat& operator*=(const T&) { return *this; }
    template<typename T> __both__ BigFloat& operator/=(const T&) { return *this; }

    __both__ static unsigned default_precision() { return 0; }
    __both__ static void default_precision(unsigned) {}
};

BigFloat operator+(double, const BigFloat&) { return {}; }
BigFloat operator-(double, const BigFloat&) { return {}; }
BigFloat operator*(double, const BigFloat&) { return {}; }
BigFloat operator/(double, const BigFloat&) { return {}; }

#else

#include <boost/multiprecision/mpfr.hpp>
using namespace boost::multiprecision;
using BigFloat = mpfr_float;

#endif

using BigComplex = Complex<BigFloat>;
