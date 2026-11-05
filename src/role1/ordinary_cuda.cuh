#pragma once
// ============================================================
// ordinary_cuda.cuh  —  Ordinary CUDA runner
//
// Launches the shared jacobi_update kernel exactly N times in
// one CUDA stream.  No per-iteration synchronization.
// ============================================================

#include <cuda_runtime.h>

// Result struct returned to the selector and command-line driver.
struct OrdinaryResult
{
    float* d_final;     // pointer to the correct GPU buffer (d_A or d_B)
                        // caller must NOT free this; it aliases one of the
                        // two buffers passed in.
    float  gpu_ms;      // GPU-event elapsed time (kernel launches only)
    float  host_ms;     // host wall-clock elapsed (same interval, for comparison)
};

// ---------------------------------------------------------------
// ordinary_cuda_run
//
// Executes N Jacobi iterations using ordinary kernel launches.
//
// Parameters:
//   d_A, d_B  - two pre-allocated GPU buffers (rows*cols floats each)
//               BOTH must already contain the same initial grid.
//               Iteration 1 reads d_A → writes d_B.
//               Iteration 2 reads d_B → writes d_A.  Etc.
//   rows, cols - grid dimensions including boundaries
//   N          - exact number of iterations
//   stream     - CUDA stream to use (must be pre-created)
//
// Returns:
//   OrdinaryResult::d_final points to d_A (even N) or d_B (odd N).
//
// After the call the stream is fully synchronized.
// The GPU buffers are NOT freed; the caller owns them.
// ---------------------------------------------------------------
OrdinaryResult ordinary_cuda_run(float* d_A, float* d_B,
                                  int rows, int cols, int N,
                                  cudaStream_t stream);

// ---------------------------------------------------------------
// ordinary_cuda_run_raw
//
// Same as ordinary_cuda_run but does NOT synchronize at the end.
// Used during CUDA Graph capture: call this inside a capture
// window only if you need ordinary launches inside a graph.
//
// Caller is responsible for synchronization.
// Returns the pointer to the final buffer (same parity rule).
// ---------------------------------------------------------------
float* ordinary_cuda_run_raw(float* d_A, float* d_B,
                              int rows, int cols, int N,
                              cudaStream_t stream);
