#pragma once

#include <cstdio>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "gpu assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
