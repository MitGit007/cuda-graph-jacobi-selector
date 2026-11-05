// ============================================================
// ordinary_cuda.cu  —  Ordinary CUDA runner
// ============================================================

#include "ordinary_cuda.cuh"
#include "jacobi_kernel.cuh"
#include "../common.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <chrono>
#include <string>

// ------------------------------------------------------------
// Internal CUDA error-check macro
// ------------------------------------------------------------
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess) {                                           \
            char buf[256];                                                 \
            std::snprintf(buf, sizeof(buf),                                \
                "CUDA error %s at %s:%d — %s",                            \
                cudaGetErrorName(_e), __FILE__, __LINE__,                  \
                cudaGetErrorString(_e));                                   \
            throw std::runtime_error(buf);                                 \
        }                                                                  \
    } while (0)

// ------------------------------------------------------------
// Internal helper: submit N kernel launches into stream.
// Returns pointer to the final buffer (no sync).
// ------------------------------------------------------------
static float* launch_n(float* d_A, float* d_B,
                        int rows, int cols, int N,
                        cudaStream_t stream)
{
    dim3 block, grid;
    launch_config(rows, cols, block, grid);

    float* src = d_A;
    float* dst = d_B;

    for (int iter = 0; iter < N; ++iter)
    {
        jacobi_update<<<grid, block, 0, stream>>>(src, dst, rows, cols);

        // Swap roles for the next iteration.
        float* tmp = src;
        src = dst;
        dst = tmp;
    }

    // After N iterations, src points to the buffer holding the result:
    //   N even → src == d_A   (same as start)
    //   N odd  → src == d_B
    return src;
}

// ------------------------------------------------------------
// ordinary_cuda_run  (synchronized)
// ------------------------------------------------------------
OrdinaryResult ordinary_cuda_run(float* d_A, float* d_B,
                                  int rows, int cols, int N,
                                  cudaStream_t stream)
{
    // GPU event timers (device-side elapsed, for diagnostics).
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Host wall-clock start.
    auto t0 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaEventRecord(ev_start, stream));

    float* d_final = launch_n(d_A, d_B, rows, cols, N, stream);

    CUDA_CHECK(cudaEventRecord(ev_stop, stream));

    // Synchronize: wait for all GPU work to finish.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Host wall-clock stop (after sync, so GPU work is complete).
    auto t1 = std::chrono::high_resolution_clock::now();

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    float host_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    return OrdinaryResult{ d_final, gpu_ms, host_ms };
}

// ------------------------------------------------------------
// ordinary_cuda_run_raw  (no sync — for use inside graph capture
//                         windows or timed sequences)
// ------------------------------------------------------------
float* ordinary_cuda_run_raw(float* d_A, float* d_B,
                              int rows, int cols, int N,
                              cudaStream_t stream)
{
    return launch_n(d_A, d_B, rows, cols, N, stream);
}
