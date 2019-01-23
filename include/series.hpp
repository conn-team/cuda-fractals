#pragma once

#include "cuda_helper.hpp"

template<typename T>
struct CubicSeries {
    T data[3];

    CubicSeries() {}
    CubicSeries(T a, T b, T c) : data{a, b, c} {}

    T& operator[](int i) { return data[i]; }

    __host__ T evalHost(T x) const {
        return ((data[2]*x + data[1])*x + data[0])*x;
    }

    __device__ T eval(T x) const {
        return ((data[2]*x + data[1])*x + data[0])*x;
    }
};
