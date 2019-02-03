#pragma once

#include <cstdio>
#include <vector>

#if (CUDART_VERSION < 9000)
#define __syncwarp()
#endif

constexpr int WARP_SIZE = 32;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "gpu assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#ifdef __CUDACC__
#define __both__  __host__ __device__
#else
#define __both__
#endif

template<typename T>
class CudaArray {
public:
    CudaArray() {}
    CudaArray(size_t n) { resizeDiscard(n); }
    CudaArray(const std::vector<T>& vec) { assign(vec); }
    CudaArray(const CudaArray& other) = delete;
    CudaArray& operator=(const CudaArray& other) = delete;
    
    ~CudaArray() {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    void resizeDiscard(size_t n) {
        if (cap < n) {
            if (ptr) {
                cudaFree(ptr);
            }
            cap = n;
            gpuErrchk(cudaMalloc(&ptr, cap*sizeof(T)));
        }
        len = n;
    }

    void assign(const std::vector<T>& vec) {
        resizeDiscard(vec.size());
        if (len > 0) {
            gpuErrchk(cudaMemcpy(ptr, vec.data(), vec.size()*sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    void get(std::vector<T>& vec) const {
        vec.resize(len);
        gpuErrchk(cudaMemcpy(vec.data(), ptr, len*sizeof(T), cudaMemcpyDeviceToHost));
    }

    T*       data()           { return ptr; }
    const T* data()     const { return ptr; }
    size_t   size()     const { return len; }
    size_t   capacity() const { return cap; }

    T get(size_t i) const {
        T ret;
        gpuErrchk(cudaMemcpy(&ret, ptr+i, sizeof(T), cudaMemcpyDeviceToHost));
        return ret;
    }

    void swap(CudaArray& other) {
        std::swap(ptr, other.ptr);
        std::swap(len, other.len);
        std::swap(cap, other.cap);
    }

private:
    T *ptr{nullptr};
    size_t len{0}, cap{0};
};

template<typename T>
class CudaVar {
public:
    CudaVar() {
        gpuErrchk(cudaMalloc(&ptr, sizeof(T)));
    }

    CudaVar(const T& elem) : CudaVar() {
        set(elem);
    }

    ~CudaVar() {
        cudaFree(ptr);
    }

    CudaVar(const CudaVar& other) = delete;
    CudaVar& operator=(const CudaVar& other) = delete;

    void set(const T& elem) {
        gpuErrchk(cudaMemcpy(ptr, &elem, sizeof(T), cudaMemcpyHostToDevice));
    }

    T get() const {
        T ret;
        gpuErrchk(cudaMemcpy(&ret, ptr, sizeof(T), cudaMemcpyDeviceToHost));
        return ret;
    }

    T *pointer() { return ptr; }

    void swap(CudaVar& other) {
        std::swap(ptr, other.ptr);
    }

private:
    T *ptr;
};
