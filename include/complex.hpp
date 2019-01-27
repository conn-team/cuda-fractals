#pragma once

#include <iostream>
#include "cuda_helper.hpp"
#include "bignum.hpp"
#include "ext_float.hpp"

template<typename T>
struct Complex {
    T x, y;

    __both__ Complex(T x = 0, T y = 0) : x(x), y(y) {}

    template<typename S>
    __both__ explicit Complex(const Complex<S>& other) : Complex(T(other.x), T(other.y)) {}

    __both__ Complex operator+(const Complex& r) const { return {x+r.x, y+r.y}; }
    __both__ Complex operator-(const Complex& r) const { return {x-r.x, y-r.y}; }
    __both__ Complex operator*(const T& r)       const { return {x*r, y*r}; }
    __both__ Complex operator*(const Complex& r) const { return {x*r.x-y*r.y, y*r.x+x*r.y}; }
    __both__ Complex operator/(const T& r)       const { return {x/r, y/r}; }

    __both__ Complex operator/(const Complex& r) const {
        T div = r.norm();
        return {(x*r.x+y*r.y) / div, (y*r.x-x*r.y) / div};
    }

    __both__ Complex& operator+=(const Complex& r) {
        x += r.x;
        y += r.y;
        return *this;
    }

    __both__ Complex& operator-=(const Complex& r) {
        x -= r.x;
        y -= r.y;
        return *this;
    }

    __both__ Complex& operator*=(const T& r) {
        x *= r;
        y *= r;
        return *this;
    }

    __both__ Complex& operator*=(const Complex& r) {
        *this = *this * r;
        return *this;
    }

    __both__ Complex& operator/=(const T& r) {
        x /= r;
        y /= r;
        return *this;
    }

    __both__ Complex& operator/=(const Complex& r) {
        *this = *this / r;
        return *this;
    }

    __both__ Complex sqr()  const { return {x*x-y*y, x*y*T(2)}; }
    __both__ T       norm() const { return x*x + y*y; }
};

template<typename T>
__both__ Complex<T> operator*(const T& l, const Complex<T>& r) { return r*l; }

template<typename T>
std::ostream& operator<<(std::ostream& out, const Complex<T>& r) {
    return out << r.x << "+" << r.y << "i";
}

using ExtComplex = Complex<ExtFloat>;
using BigComplex = Complex<BigFloat>;
