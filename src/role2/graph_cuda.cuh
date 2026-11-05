#pragma once
#include <cuda_runtime.h>

// Result of one CUDA Graph run. The final buffer aliases d_A or d_B;
// ownership remains with the caller.
struct GraphResult {
    float* d_final;
    float  setup_ms; // stream capture + graph instantiate
    float  exec_ms;  // graph replay(s) + leftover ordinary launches
    float  host_ms;  // same as exec_ms; retained for consistent interface
};

GraphResult run_graph(float* d_A, float* d_B,
                      int rows, int cols, int N, int batch,
                      cudaStream_t stream);
