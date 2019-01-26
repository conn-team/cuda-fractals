#pragma once

#include "cuda_helper.hpp"
#include "complex.hpp"

// Represents polynomial a*x + b*x^2 + c*x^3
template<typename T>
class CubicSeries {
public:
    __both__ CubicSeries() {}
    __both__ CubicSeries(T a, T b, T c) : data{a, b, c} {}

    __both__ T& operator[](int i) { return data[i]; }

    __both__ T evaluate(T x) const {
        return ((data[2]*x + data[1])*x + data[0])*x;
    }

private:
    T data[3];
};
