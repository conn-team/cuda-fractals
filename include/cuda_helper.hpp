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
#define __both__ _Pragma("hd_warning_disable") __host__ __device__
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
		gpuErrchk(cudaMemcpy(vec.data(), ptr, len*sizeof(T), cudaMemcpyDeviceToHost));
	}

	T*       data()           { return ptr; }
	const T* data()     const { return ptr; }
	size_t   size()     const { return len; }
	size_t   capacity() const { return cap; }

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

private:
	T *ptr;
};
