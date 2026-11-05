// ============================================================
// test_correctness.cu  —  Jacobi correctness test suite
//
// Runs 5 structured test cases comparing CPU and CUDA results.
// Also includes a hand-checkable 5x5 case and boundary checks.
//
// Build:  (via CMake — see CMakeLists.txt target: jacobi_tests)
// Run:    ./build/jacobi_tests
//
// Exit code 0 = all tests passed.
// Exit code 1 = one or more tests failed.
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdexcept>

#include <cuda_runtime.h>

#include "../src/common.h"
#include "../src/role1/cpu_jacobi.h"
#include "../src/role1/jacobi_kernel.cuh"
#include "../src/role1/ordinary_cuda.cuh"

// ---- CUDA error check --------------------------------------
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

// ---- Test runner -------------------------------------------
struct TestResult {
    const char* name;
    int   rows, cols, N;
    bool  pass;
    float max_err;
};

// Run one test: generate input, run CPU + GPU, compare.
static TestResult run_test(const char* name,
                            int rows, int cols, int N,
                            unsigned seed = 42)
{
    std::printf("  %-40s  rows=%-4d cols=%-4d N=%-4d  ... ", name, rows, cols, N);
    std::fflush(stdout);

    const int total = rows * cols;

    // Generate initial grid
    float* h_init = generate_input(rows, cols, seed);

    // CPU reference
    float* h_cpu = new float[total];
    cpu_jacobi(h_init, h_cpu, rows, cols, N);

    // GPU ordinary
    float* d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, total * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A, h_init, total * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_init, total * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    OrdinaryResult res = ordinary_cuda_run(d_A, d_B, rows, cols, N, stream);

    float* h_gpu = new float[total];
    CUDA_CHECK(cudaMemcpy(h_gpu, res.d_final, total * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare
    float max_err = 0.0f;
    bool pass = check_correctness(h_cpu, h_gpu, rows, cols, &max_err, false);

    std::printf("%s  (max_err = %.3e)\n", pass ? "PASS" : "FAIL", max_err);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    delete[] h_init;
    delete[] h_cpu;
    delete[] h_gpu;

    return { name, rows, cols, N, pass, max_err };
}

// ---- Hand-check: 5x5, all-zero interior, N=1 ---------------
// Top boundary = 1 (excluding corners), rest = 0.
// Expected: cell (1,1) = (1+0+0+0)/4 = 0.25
//           cell (1,2) = (1+0+0+0)/4 = 0.25
//           cell (1,3) = (1+0+0+0)/4 = 0.25
// (the interior of a 5x5 grid is 3x3)
static bool hand_check_5x5()
{
    std::printf("  %-40s  rows=5    cols=5    N=1     ... ", "Hand-check 5x5 zero interior");
    std::fflush(stdout);

    const int rows = 5, cols = 5, N = 1, total = 25;

    // Build the grid manually: all zeros, top boundary = 1.
    float h_init[25] = {};
    apply_boundaries(h_init, rows, cols);  // sets top row [1..3] = 1

    // CPU
    float h_cpu[25] = {};
    cpu_jacobi(h_init, h_cpu, rows, cols, N);

    // GPU
    float* d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, total * sizeof(float)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpy(d_A, h_init, total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_init, total * sizeof(float), cudaMemcpyHostToDevice));

    OrdinaryResult res = ordinary_cuda_run(d_A, d_B, rows, cols, N, stream);

    float h_gpu[25] = {};
    CUDA_CHECK(cudaMemcpy(h_gpu, res.d_final, total * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    // Hand-verify: row 1 (first interior row), cols 1-3 each should be 0.25
    // Row 2 (middle interior) has zero neighbours all around after 1 iter → stays 0 conceptually
    // but cell (2,1): north=h_init[1*5+1]=0, south=h_init[3*5+1]=0, west=h_init[2*5+0]=0,
    //                 east=h_init[2*5+2]=0  → 0.0  ✓
    bool ok = true;
    // Check the three directly-below-top interior cells
    for (int j = 1; j <= 3; ++j) {
        float expected = 0.25f;  // (top=1 + 0 + 0 + 0) / 4
        float got_cpu = h_cpu[IDX(1, j, cols)];
        float got_gpu = h_gpu[IDX(1, j, cols)];
        if (std::fabs(got_cpu - expected) > 1e-6f ||
            std::fabs(got_gpu - expected) > 1e-6f) {
            ok = false;
            std::printf("\n    Expected (1,%d)=0.25  CPU=%.6f  GPU=%.6f", j, got_cpu, got_gpu);
        }
    }
    // Check boundaries unchanged
    float max_err = 0.0f;
    bool boundary_ok = check_correctness(h_cpu, h_gpu, rows, cols, &max_err, false);
    ok = ok && boundary_ok;

    std::printf("%s  (stencil check + CPU vs GPU match)\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ---- Buffer alternation check: 8x8, N=1,2,3 ----------------
static bool buffer_alternation_test()
{
    std::printf("  %-40s  rows=8    cols=8    N=1,2,3  ... ", "Buffer alternation (N=1,2,3)");
    std::fflush(stdout);

    const int rows = 8, cols = 8, total = 64;
    bool ok = true;

    float* h_init = generate_input(rows, cols, 42);

    for (int N : {1, 2, 3}) {
        float* h_cpu = new float[total];
        cpu_jacobi(h_init, h_cpu, rows, cols, N);

        float* d_A, *d_B;
        CUDA_CHECK(cudaMalloc(&d_A, total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, total * sizeof(float)));
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        CUDA_CHECK(cudaMemcpy(d_A, h_init, total * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_init, total * sizeof(float), cudaMemcpyHostToDevice));

        OrdinaryResult res = ordinary_cuda_run(d_A, d_B, rows, cols, N, stream);

        // Verify correct buffer chosen (even N → d_A, odd N → d_B)
        float* expected_buf = (N % 2 == 0) ? d_A : d_B;
        if (res.d_final != expected_buf) {
            std::printf("\n    N=%d: wrong final buffer", N);
            ok = false;
        }

        float* h_gpu = new float[total];
        CUDA_CHECK(cudaMemcpy(h_gpu, res.d_final, total * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        if (!check_correctness(h_cpu, h_gpu, rows, cols, &max_err, false)) {
            std::printf("\n    N=%d: correctness FAIL (max_err=%.3e)", N, max_err);
            ok = false;
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        delete[] h_cpu;
        delete[] h_gpu;
    }

    delete[] h_init;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ============================================================
// main
// ============================================================
int main()
{
    std::printf("=== Jacobi Correctness Tests ===\n\n");

    // Print GPU info
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("GPU: %s  (sm_%d%d)\n\n", prop.name,
                prop.major, prop.minor);

    bool all_pass = true;

    // ---- Structured test cases (plan §4) -------------------
    std::printf("--- Structured Tests ---\n");

    TestResult cases[] = {
        run_test("T1: 5x5, random interior, N=1",      5,  5,   1),
        run_test("T2a: 8x8, N=1",                      8,  8,   1),
        run_test("T2b: 8x8, N=2",                      8,  8,   2),
        run_test("T2c: 8x8, N=3",                      8,  8,   3),
        run_test("T3: 17x19 rectangular, N=33",        17, 19,  33),
        run_test("T4: 32x32, N=100",                   32, 32,  100),
        run_test("T5: 64x64, N=511 (odd, long)",       64, 64,  511),
    };

    std::printf("\n--- Hand-check Tests ---\n");
    bool hc  = hand_check_5x5();
    bool alt = buffer_alternation_test();

    // ---- Summary -------------------------------------------
    std::printf("\n=== Summary ===\n");
    std::printf("%-40s  %6s  %12s\n", "Test", "Result", "Max Error");
    std::printf("%-40s  %6s  %12s\n",
                "----------------------------------------",
                "------", "------------");

    for (auto& r : cases) {
        std::printf("%-40s  %6s  %12.3e\n",
                    r.name, r.pass ? "PASS" : "FAIL", r.max_err);
        if (!r.pass) all_pass = false;
    }
    std::printf("%-40s  %6s\n", "Hand-check 5x5",       hc  ? "PASS" : "FAIL");
    std::printf("%-40s  %6s\n", "Buffer alternation",    alt ? "PASS" : "FAIL");
    if (!hc || !alt) all_pass = false;

    std::printf("\nOverall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}
