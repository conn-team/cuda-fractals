#pragma once

#include "cuda_helper.hpp"

template<typename T>
struct Complex {
    T x, y;

    __both__ Complex(T x = 0, T y = 0) : x(x), y(y) {}

    __both__ Complex operator+(const Complex& r) const { return {x+r.x, y+r.y}; }
    __both__ Complex operator-(const Complex& r) const { return {x-r.x, y-r.y}; }
    __both__ Complex operator*(const T& r)       const { return {x*r, y*r}; }
    __both__ Complex operator*(const Complex& r) const { return {x*r.x-y*r.y, y*r.x+x*r.y}; }

    __both__ Complex sqr() const { return {x*x-y*y, 2*x*y}; }

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

    __both__ T norm() const {
        return x*x + y*y;
    }

    __both__ T dot(const Complex& r) const {
        return x*r.x + y*r.y;
    }
};

template<typename T>
__both__ Complex<T> operator*(const T& l, const Complex<T>& r) { return r*l; }

using DevComplex = Complex<double>;
