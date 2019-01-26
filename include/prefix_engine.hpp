#pragma once

#include <cstdint>

#include "cuda_helper.hpp"

namespace prefix_engine {
    constexpr int BLOCK_SIZE = 1024;
    constexpr int TOTAL_SEGMENTS = WARP_SIZE;

    template<typename T, typename F>
    __device__ void aggregateWarp(T *data) {
        int id = threadIdx.x % WARP_SIZE;
        T val = data[id];

        if (id >= 1) { val = F{}(data[id-1], val); }
        __syncwarp();
        data[id] = val;
        __syncwarp();

        if (id >= 2) { val = F{}(data[id-2], val); }
        __syncwarp();
        data[id] = val;
        __syncwarp();

        if (id >= 4) { val = F{}(data[id-4], val); }
        __syncwarp();
        data[id] = val;
        __syncwarp();

        if (id >= 8) { val = F{}(data[id-8], val); }
        __syncwarp();
        data[id] = val;
        __syncwarp();

        if (id >= 16) { val = F{}(data[id-16], val); }
        __syncwarp();
        data[id] = val;
    }

    template<typename T, typename F>
    __device__ void aggregateBlock(T *data, int size, T &toAdd, bool hasAdd) {
        __shared__ T shData[BLOCK_SIZE], warpSums[WARP_SIZE];

        int id = threadIdx.x;
        int warpId = id / WARP_SIZE;
        int warpOffset = warpId * WARP_SIZE;

        if (id < size) {
            if (hasAdd && id == 0) {
                shData[id] = F{}(toAdd, data[id]);
            } else {
                shData[id] = data[id];
            }
        }

        __syncwarp();
        aggregateWarp<T, F>(shData+warpOffset);

        if (id % WARP_SIZE == WARP_SIZE-1) {
            warpSums[warpId] = shData[id];
        }

        __syncthreads();

        if (id < WARP_SIZE) {
            aggregateWarp<T, F>(warpSums);
        }

        __syncthreads();

        if (id < size) {
            if (warpId > 0) {
                data[id] = F{}(warpSums[warpId-1], shData[id]);
            } else {
                data[id] = shData[id];
            }
        }

        toAdd = warpSums[WARP_SIZE-1];
    }

    template<typename T, typename F>
    __global__ void aggregateSegments(T *data, T *segmentSums, int blocksPerSegment, int size) {
        int from = blockIdx.x * blocksPerSegment * BLOCK_SIZE;
        int to = min(from + blocksPerSegment*BLOCK_SIZE, size);
        T toAdd;

        for (int i = from; i < to; i += BLOCK_SIZE) {
            aggregateBlock<T, F>(data+i, size-i, toAdd, i > from);
        }

        segmentSums[blockIdx.x] = toAdd;
    }

    template<typename T, typename F>
    __global__ void aggregateSegmentSums(T *data) {
        aggregateWarp<T, F>(data);
    }

    template<typename T, typename F>
    __global__ void addSegmentSums(T *data, T *segmentSums, int blocksPerSegment, int size) {
        int segment = blockIdx.x / blocksPerSegment;
        int i = blockIdx.x*BLOCK_SIZE + threadIdx.x;

        if (i < size) {
            data[i] = F{}(segmentSums[segment], data[i]);
        }
    }
};

template<typename T, typename F>
void prefixAggregate(CudaArray<T> &data) {
    using namespace prefix_engine;

    int n = data.size();
    int nBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocksPerSegment = (nBlocks + TOTAL_SEGMENTS - 1) / TOTAL_SEGMENTS;
    int totalSegments = std::min(nBlocks, TOTAL_SEGMENTS);
    int blockSize = std::min(n, BLOCK_SIZE);
    int segmentSize = blocksPerSegment*BLOCK_SIZE;

    CudaArray<T> segmentSums(TOTAL_SEGMENTS);
    aggregateSegments<T, F><<<totalSegments, blockSize>>>(data.data(), segmentSums.data(), blocksPerSegment, n);

    if (totalSegments > 1) {
        aggregateSegmentSums<T, F><<<1, TOTAL_SEGMENTS>>>(segmentSums.data());
        addSegmentSums<T, F><<<nBlocks-blocksPerSegment, BLOCK_SIZE>>>(data.data()+segmentSize, segmentSums.data(), blocksPerSegment, n-segmentSize);
    }
}
