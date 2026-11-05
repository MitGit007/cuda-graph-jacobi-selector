// ============================================================
// main.cu  —  Unified CLI entry point for the Jacobi solver
//
// Usage:
//   jacobi --mode <cpu|ordinary|graph|selector>
//          --rows R --cols C --iters N --seed S
//          [--batch B]          # required for --mode graph
//          [--output path.bin]  # save final grid as raw float32
//          [--csv   path.csv]   # append one timing record
//
// Every mode:
//   1. Generates the initial grid from seed on the CPU.
//   2. Copies it to both GPU buffers d_A and d_B.
//   3. Runs the requested mode.
//   4. Copies the final grid back to the CPU.
//   5. Prints a one-line summary and optional timing.
//
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <chrono>
#include <limits>

#include <cuda_runtime.h>

#include "common.h"
#include "role1/cpu_jacobi.h"
#include "role1/jacobi_kernel.cuh"
#include "role1/ordinary_cuda.cuh"
#include "role2/graph_cuda.cuh"
#include "role3/selector.cuh"

// ---- CUDA error helper -------------------------------------
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d — %s\n",        \
                cudaGetErrorName(_e), __FILE__, __LINE__,                  \
                cudaGetErrorString(_e));                                   \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

// ---- Argument parsing --------------------------------------
struct Args {
    std::string mode;
    int   rows   = 128;
    int   cols   = 128;
    int   iters  = 100;
    int   seed   = 42;
    int   batch  = 0;
    std::string output;
    std::string csv;
};

static void print_usage(const char* prog)
{
    std::printf("Usage: %s --mode <cpu|ordinary|graph|selector>"
                " --rows R --cols C --iters N --seed S"
                " [--batch B] [--output file.bin] [--csv file.csv]\n", prog);
}

static Args parse_args(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") { print_usage(argv[0]); std::exit(0); }
        if (i + 1 >= argc) { print_usage(argv[0]); std::exit(1); }
        std::string val = argv[++i];
        if      (key == "--mode")   a.mode   = val;
        else if (key == "--rows")   a.rows   = std::stoi(val);
        else if (key == "--cols")   a.cols   = std::stoi(val);
        else if (key == "--iters")  a.iters  = std::stoi(val);
        else if (key == "--seed")   a.seed   = std::stoi(val);
        else if (key == "--batch")  a.batch  = std::stoi(val);
        else if (key == "--output") a.output = val;
        else if (key == "--csv")    a.csv    = val;
        else { std::fprintf(stderr, "Unknown argument: %s\n", key.c_str()); std::exit(1); }
    }
    if (a.mode.empty()) { print_usage(argv[0]); std::exit(1); }
    return a;
}

// ---- CSV helper --------------------------------------------
static void append_csv(const std::string& path,
                        const std::string& mode,
                        int rows, int cols, int iters, int seed, int batch,
                        int correctness_status, float max_err,
                        float calib_ms, float setup_ms,
                        float exec_ms,  float total_ms)
{
    bool write_header = false;
    {
        FILE* f = std::fopen(path.c_str(), "r");
        if (!f) write_header = true;
        else    std::fclose(f);
    }
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) { std::fprintf(stderr, "Cannot open CSV: %s\n", path.c_str()); return; }
    if (write_header)
        std::fprintf(f, "rows,cols,iterations,seed,mode,batch_size,"
                        "correctness_status,max_error,"
                        "calibration_ms,setup_ms,execution_ms,total_ms\n");
    std::fprintf(f, "%d,%d,%d,%d,%s,%d,%d,%.6e,%.3f,%.3f,%.3f,%.3f\n",
                 rows, cols, iters, seed,
                 mode.c_str(), batch,
                 correctness_status, max_err,
                 calib_ms, setup_ms, exec_ms, total_ms);
    std::fclose(f);
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv)
{
    Args a = parse_args(argc, argv);

    std::printf("=== Jacobi Solver ===\n");
    std::printf("Mode: %s | Grid: %dx%d | Iters: %d | Seed: %d\n",
                a.mode.c_str(), a.rows, a.cols, a.iters, a.seed);
    if (a.mode == "graph")
        std::printf("Batch: %d\n", a.batch);
    std::printf("\n");

    // ---- 1. Generate initial grid on the CPU ---------------
    float* h_init = nullptr;
    try {
        h_init = generate_input(a.rows, a.cols, (unsigned)a.seed);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Input generation failed: %s\n", e.what());
        return 1;
    }

    const int total = a.rows * a.cols;

    // ---- 2. CPU mode ---------------------------------------
    if (a.mode == "cpu") {
        float* h_out = new float[total];
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_jacobi(h_init, h_out, a.rows, a.cols, a.iters);
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        std::printf("CPU done in %.3f ms\n", ms);

        if (!a.output.empty()) {
            save_grid(a.output.c_str(), h_out, a.rows, a.cols);
            std::printf("Saved to %s\n", a.output.c_str());
        }
        if (!a.csv.empty())
            append_csv(a.csv, "cpu", a.rows, a.cols, a.iters, a.seed, 0,
                       1, 0.0f, 0.0f, 0.0f, ms, ms);

        delete[] h_out;
        delete[] h_init;
        return 0;
    }

    // ---- 3. GPU modes: allocate buffers and stream ---------
    float* d_A = nullptr;
    float* d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, total * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Report allocation (common context + buffers)
    std::printf("GPU buffers: 2 x %d floats = %.2f MiB\n",
                total, 2.0f * total * sizeof(float) / (1024.0f * 1024.0f));

    // Host buffer for result
    float* h_result = new float[total];

    // Timing
    float calib_ms = 0.0f, setup_ms = 0.0f, exec_ms = 0.0f, total_ms = 0.0f;
    float* d_final = nullptr;
    // 1 = checked and passed, 0 = checked and failed, -1 = not checked here.
    int correctness_status = -1;
    float max_err = std::numeric_limits<float>::quiet_NaN();

    // Wall-clock around the ENTIRE solver (includes data transfer)
    auto wall_start = std::chrono::high_resolution_clock::now();

    // ---- 4a. Copy initial grid to BOTH GPU buffers ---------
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_init, total * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_init, total * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ---- 4b. Run requested mode ----------------------------
    if (a.mode == "ordinary")
    {
        OrdinaryResult res = ordinary_cuda_run(d_A, d_B, a.rows, a.cols, a.iters, stream);
        d_final  = res.d_final;
        exec_ms  = res.host_ms;
        setup_ms = 0.0f;
    }
    else if (a.mode == "graph")
    {
        if (a.batch <= 0) {
            std::fprintf(stderr, "Error: --mode graph requires --batch B (B > 0)\n");
            return 1;
        }
        GraphResult res = run_graph(d_A, d_B, a.rows, a.cols, a.iters, a.batch, stream);
        d_final  = res.d_final;
        setup_ms = res.setup_ms;
        exec_ms  = res.exec_ms;
    }
    else if (a.mode == "selector")
{
    SelectorResult res = run_selector(
        d_A, d_B, h_init, a.rows, a.cols, a.iters, stream);

    d_final  = res.d_final;
    calib_ms = res.calib_ms;
    setup_ms = res.setup_ms;
    exec_ms  = res.exec_ms;
    a.batch  = res.chosen_batch;

    std::printf("Selector chose: %s\n",
                res.chosen_batch < 0 ? "ordinary" :
                ("batch=" + std::to_string(res.chosen_batch)).c_str());
}
    else {
        std::fprintf(stderr, "Unknown mode: %s\n", a.mode.c_str());
        return 1;
    }

    // ---- 5. Copy result back to CPU and sync ---------------
    CUDA_CHECK(cudaMemcpy(h_result, d_final, total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto wall_end = std::chrono::high_resolution_clock::now();
    total_ms = std::chrono::duration<float, std::milli>(wall_end - wall_start).count();

    // ---- 6. Correctness check (compare against CPU) --------
    // Large grids are validated separately against the ordinary CUDA output.
    const long long CHECK_LIMIT = 512LL * 512LL;
    if ((long long)total <= CHECK_LIMIT) {
        float* h_ref = new float[total];
        cpu_jacobi(h_init, h_ref, a.rows, a.cols, a.iters);
        const bool correct =
            check_correctness(h_ref, h_result, a.rows, a.cols, &max_err, true);
        correctness_status = correct ? 1 : 0;
        std::printf("Correctness vs CPU: %s  (max err = %.3e)\n",
                    correct ? "PASS" : "FAIL", max_err);
        delete[] h_ref;
    } else {
        std::printf("Correctness status: NOT_CHECKED in this timing run "
                    "(use scripts\\validate_large.ps1).\n");
    }

    // ---- 7. Save output ------------------------------------
    if (!a.output.empty()) {
        save_grid(a.output.c_str(), h_result, a.rows, a.cols);
        std::printf("Final grid saved to %s\n", a.output.c_str());
    }

    // ---- 8. Print timing summary ---------------------------
    std::printf("Timing: calib=%.3f ms  setup=%.3f ms  exec=%.3f ms  total=%.3f ms\n",
                calib_ms, setup_ms, exec_ms, total_ms);

    // ---- 9. Append CSV record ------------------------------
    if (!a.csv.empty())
        append_csv(a.csv, a.mode, a.rows, a.cols, a.iters, a.seed, a.batch,
                   correctness_status, max_err,
                   calib_ms, setup_ms, exec_ms, total_ms);

    // ---- Cleanup -------------------------------------------
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    delete[] h_result;
    delete[] h_init;

    return correctness_status == 0 ? 2 : 0;
    // Exit code 2 means a performed correctness check failed.
}

