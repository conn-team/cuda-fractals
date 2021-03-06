#pragma once

#include "cuda_helper.hpp"
#include "complex.hpp"

constexpr int SERIES_DEGREE = 32;

template<typename T>
class Series {
public:
    __both__ Series() {}

    template<typename ...Args>
    __both__ Series(const Args&... args) : data{args...} {}

    __both__ const T& operator[](int i) const { return data[i]; }
    __both__ T& operator[](int i) { return data[i]; }

    __both__ T evaluate(T x) const {
        T ret(0);
        for (int i = SERIES_DEGREE-1; i >= 0; i--) {
            ret = (ret + data[i]) * x;
        }
        return ret;
    }

private:
    T data[SERIES_DEGREE];
};
