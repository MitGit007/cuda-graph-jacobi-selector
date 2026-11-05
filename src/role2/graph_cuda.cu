// ============================================================
// graph_cuda.cu — CUDA Graph batching for Jacobi
// ============================================================
#include "graph_cuda.cuh"
#include "../role1/jacobi_kernel.cuh"
#include "../common.h"

#include <chrono>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) do { \
    cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error ") + \
            cudaGetErrorName(e_) + " at " + __FILE__ + ":" + \
            std::to_string(__LINE__) + " — " + cudaGetErrorString(e_)); \
    } \
} while (0)

static float* launch_remainder(float* d_A, float* d_B,
                               int rows, int cols, int count,
                               float* current, cudaStream_t stream)
{
    dim3 block, grid;
    launch_config(rows, cols, block, grid);
    float* src = current;
    float* dst = (current == d_A) ? d_B : d_A;
    for (int i = 0; i < count; ++i) {
        jacobi_update<<<grid, block, 0, stream>>>(src, dst, rows, cols);
        CUDA_CHECK(cudaPeekAtLastError());
        float* tmp = src; src = dst; dst = tmp;
    }
    return src;
}

GraphResult run_graph(float* d_A, float* d_B,
                      int rows, int cols, int N, int batch,
                      cudaStream_t stream)
{
    if (!d_A || !d_B) throw std::invalid_argument("run_graph: buffers must not be null");
    if (rows < 3 || cols < 3) throw std::invalid_argument("run_graph: grid must be at least 3x3");
    if (N < 0) throw std::invalid_argument("run_graph: iteration count must be non-negative");
    if (batch <= 0) throw std::invalid_argument("run_graph: batch must be positive");
    if (batch > N && N != 0)
        throw std::invalid_argument("run_graph: batch must be <= N");
    if (N == 0)
        return {d_A, 0.0f, 0.0f, 0.0f};

    // This project evaluates the fixed candidate set, all of which are even.
    bool valid_batch = false;
    for (int i = 0; i < NUM_CANDIDATES; ++i)
        if (batch == BATCH_CANDIDATES[i]) valid_batch = true;
    if (!valid_batch)
        throw std::invalid_argument("run_graph: batch must be one of 2,4,8,16,32,64,100,128");
    if (batch > N)
        throw std::invalid_argument("run_graph: batch must be <= N");

    dim3 block, grid;
    launch_config(rows, cols, block, grid);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    auto setup_start = std::chrono::steady_clock::now();

    // Capture exactly one batch. All candidate B values are even, so the
    // captured sequence starts and ends on d_A and is safe to replay.
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    float* src = d_A;
    float* dst = d_B;
    for (int i = 0; i < batch; ++i) {
        jacobi_update<<<grid, block, 0, stream>>>(src, dst, rows, cols);
        CUDA_CHECK(cudaPeekAtLastError());
        float* tmp = src; src = dst; dst = tmp;
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    if (!graph) throw std::runtime_error("CUDA Graph capture returned a null graph");
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, 0));
    auto setup_end = std::chrono::steady_clock::now();
    float setup_ms = std::chrono::duration<float, std::milli>(
        setup_end - setup_start).count();

    const int full_replays = N / batch;
    const int remainder = N % batch;

    auto exec_start = std::chrono::steady_clock::now();
    for (int i = 0; i < full_replays; ++i)
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));

    // Even batch sizes preserve the starting-buffer parity after each replay.
    float* d_final = launch_remainder(d_A, d_B, rows, cols, remainder, d_A, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto exec_end = std::chrono::steady_clock::now();

    float exec_ms = std::chrono::duration<float, std::milli>(
        exec_end - exec_start).count();

    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));

    return {d_final, setup_ms, exec_ms, exec_ms};
}
