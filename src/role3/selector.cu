#include "selector.cuh"
#include "selector.hpp"
#include "../role1/ordinary_cuda.cuh"
#include "../role2/graph_cuda.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess)                                        \
        throw std::runtime_error(std::string(#call) + ": " +       \
                                 cudaGetErrorString(err));         \
} while (0)

static void restore_input(float* d_A, float* d_B, const float* h_init,
                          int rows, int cols, cudaStream_t stream)
{
    const size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);
    CHECK_CUDA(cudaMemcpyAsync(d_A, h_init, bytes,
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_B, h_init, bytes,
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

static SelectorResult run_ordinary(float* d_A, float* d_B,
                                   int rows, int cols, int N,
                                   cudaStream_t stream)
{
    OrdinaryResult result =
        ordinary_cuda_run(d_A, d_B, rows, cols, N, stream);
    return {result.d_final, -1, 0.0f, 0.0f, result.host_ms};
}

SelectorResult run_selector(
    float* d_A, float* d_B, const float* h_init,
    int rows, int cols, int N, cudaStream_t stream)
{
    using Clock = std::chrono::steady_clock;

    if (!h_init) throw std::invalid_argument("Null initial grid");

    // Frozen after inspecting development grids only. Short solves and
    // compute-heavy medium-length solves could not amortize calibration.
    constexpr int kShortSolveLimit = 256;
    constexpr long long kLargeGridCells = 512LL * 512LL;
    constexpr int kLargeGridCalibrationStart = 4096;
    const long long cells = static_cast<long long>(rows) * cols;

    if (N < kShortSolveLimit ||
        (cells >= kLargeGridCells && N < kLargeGridCalibrationStart)) {
        return run_ordinary(d_A, d_B, rows, cols, N, stream);
    }

    const auto batches = jacobi_selection::expected_probe_batches(N);
    if (batches.empty()) {
        return run_ordinary(d_A, d_B, rows, cols, N, stream);
    }

    auto calibration_start = Clock::now();

    const int probe_iterations = std::min(N, 16);
    OrdinaryResult ordinary_probe =
        ordinary_cuda_run(d_A, d_B, rows, cols, probe_iterations, stream);
    const double ordinary_ms_per_iteration =
        ordinary_probe.host_ms / probe_iterations;
    restore_input(d_A, d_B, h_init, rows, cols, stream);

    std::vector<jacobi_selection::Probe> probes;
    for (int batch : batches) {
        GraphResult probe =
            run_graph(d_A, d_B, rows, cols, batch, batch, stream);
        probes.push_back({batch, probe.setup_ms, probe.exec_ms, false});
        restore_input(d_A, d_B, h_init, rows, cols, stream);
    }

    const auto decision = jacobi_selection::select_policy(
        N, ordinary_ms_per_iteration, probes);

    auto calibration_end = Clock::now();
    const float calibration_ms =
        std::chrono::duration<float, std::milli>(
            calibration_end - calibration_start).count();

    if (!decision.use_graph) {
        OrdinaryResult result =
            ordinary_cuda_run(d_A, d_B, rows, cols, N, stream);
        return {result.d_final, -1, calibration_ms, 0.0f, result.host_ms};
    }

    GraphResult result =
        run_graph(d_A, d_B, rows, cols, N, decision.batch, stream);
    return {result.d_final, decision.batch, calibration_ms,
            result.setup_ms, result.exec_ms};
}
