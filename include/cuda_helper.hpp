#pragma once

#include <cstdio>
#include <vector>

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
#define __both__ __host__ __device__
#else
#define __both__
#endif

template<typename T>
class CudaArray {
public:
	CudaArray() {}
	CudaArray(const std::vector<T>& vec) { assign(vec); }
	CudaArray(const CudaArray& other) = delete;
	CudaArray& operator=(const CudaArray& other) = delete;
	
	~CudaArray() {
		if (ptr) {
			cudaFree(ptr);
		}
	}

	void assign(const std::vector<T>& vec) {
		if (!vec.empty()) {
			if (cap < vec.size()) {
				if (ptr) {
					cudaFree(ptr);
				}
				cap = vec.size();
				gpuErrchk(cudaMalloc(&ptr, cap*sizeof(T)));
			}
			gpuErrchk(cudaMemcpy(ptr, vec.data(), vec.size()*sizeof(T), cudaMemcpyHostToDevice));
		}
		len = vec.size();
	}

	void get(std::vector<T>& vec) const {
		vec.resize(len);
		gpuErrchk(cudaMemcpy(vec.data(), ptr, len*sizeof(T)));
	}

	T*       data()           { return ptr; }
	const T* data()     const { return ptr; }
	size_t   size()     const { return len; }
	size_t   capacity() const { return cap; }

private:
	T *ptr{nullptr};
	size_t len{0}, cap{0};
};
