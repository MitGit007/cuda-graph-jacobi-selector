#pragma once
#include <cuda_runtime.h>

struct SelectorResult {
    float* d_final;
    int chosen_batch;       // -1 means ordinary CUDA
    float calib_ms;
    float setup_ms;          // setup of the final chosen graph
    float exec_ms;
};

SelectorResult run_selector(
    float* d_A, float* d_B, const float* h_init,
    int rows, int cols, int N, cudaStream_t stream);