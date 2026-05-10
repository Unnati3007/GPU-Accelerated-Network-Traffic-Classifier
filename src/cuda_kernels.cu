// cuda_kernels.cu
// Custom CUDA kernels for packet feature preprocessing.
// Compiled with: nvcc -O3 -arch=sm_80 -shared -o cuda_kernels.so cuda_kernels.cu

#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

/**
 * Normalize a batch of feature vectors in-place.
 * Each thread handles one (sample, feature) element.
 * data   : [n_samples x n_features] row-major float array on GPU
 * mean   : [n_features] precomputed feature means
 * std    : [n_features] precomputed feature std devs
 */
__global__ void normalize_features(float* __restrict__ data,
                                   const float* __restrict__ mean,
                                   const float* __restrict__ std,
                                   int n_samples, int n_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_samples * n_features;
    if (idx >= total) return;

    int feat = idx % n_features;
    data[idx] = (data[idx] - mean[feat]) / (std[feat] + 1e-8f);
}

/**
 * Clip feature values to [-clip_val, clip_val] to remove outliers.
 */
__global__ void clip_features(float* __restrict__ data, float clip_val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = fmaxf(-clip_val, fminf(clip_val, data[idx]));
}

/**
 * Replace NaN and Inf values with 0.
 */
__global__ void sanitize_features(float* __restrict__ data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (!isfinite(data[idx])) data[idx] = 0.0f;
}

// Host-side launcher (called from Python via ctypes)
extern "C" {
    void launch_normalize(float* data, float* mean, float* std,
                          int n_samples, int n_features, cudaStream_t stream) {
        int total = n_samples * n_features;
        int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        normalize_features<<<blocks, BLOCK_SIZE, 0, stream>>>(
            data, mean, std, n_samples, n_features
        );
    }

    void launch_sanitize(float* data, int n, cudaStream_t stream) {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        sanitize_features<<<blocks, BLOCK_SIZE, 0, stream>>>(data, n);
    }

    void launch_clip(float* data, float clip_val, int n, cudaStream_t stream) {
        int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        clip_features<<<blocks, BLOCK_SIZE, 0, stream>>>(data, clip_val, n);
    }
}
