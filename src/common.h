#pragma once
// ============================================================
// common.h  —  Shared types, constants, and input generation
// ============================================================

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>

// --------------- Index macro (row-major) --------------------
// Grid dimensions include boundary cells.
// Interior cells: i in [1, rows-2], j in [1, cols-2].
#define IDX(i, j, cols) ((i) * (cols) + (j))

// --------------- Numerical tolerance -------------------------
// |GPU - CPU| <= REL_TOL * (1 + |CPU|)  per cell
static constexpr float REL_TOL = 1e-5f;

// --------------- Grid dimensions ----------------------------
struct GridDims {
    int rows;
    int cols;
    int total() const { return rows * cols; }
};

// --------------- Candidate batch sizes ----------------------
static constexpr int BATCH_CANDIDATES[] = {2, 4, 8, 16, 32, 64, 100, 128};
static constexpr int NUM_CANDIDATES = 8;

// --------------- Input generation ---------------------------
// Allocates and returns a CPU grid of size rows*cols (float).
// Caller must free[] the returned pointer.
//
// Layout:
//   Top row (excluding corners): 1.0f
//   All corners, bottom row, left col, right col: 0.0f
//   Interior: uniform random in [0, 1] from mt19937(seed)
//
// Both GPU buffers must be initialized from this same array.
inline float* generate_input(int rows, int cols, unsigned int seed)
{
    if (rows < 3 || cols < 3)
        throw std::invalid_argument("Grid must be at least 3x3 (includes boundaries)");

    float* grid = new float[rows * cols]();   // zero-initialised

    // Top boundary (excluding corners) = 1
    for (int j = 1; j < cols - 1; ++j)
        grid[IDX(0, j, cols)] = 1.0f;

    // Interior: reproducible pseudorandom values in [0, 1]
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 1; i < rows - 1; ++i)
        for (int j = 1; j < cols - 1; ++j)
            grid[IDX(i, j, cols)] = dist(rng);

    return grid;
}

// --------------- Boundary enforcement -----------------------
// Resets boundary cells to their canonical values.
// Call this after restoring a grid for calibration replay.
inline void apply_boundaries(float* grid, int rows, int cols)
{
    // All borders → 0 first
    for (int j = 0; j < cols; ++j) {
        grid[IDX(0,        j, cols)] = 0.0f;   // top
        grid[IDX(rows-1,   j, cols)] = 0.0f;   // bottom
    }
    for (int i = 0; i < rows; ++i) {
        grid[IDX(i, 0,      cols)] = 0.0f;     // left
        grid[IDX(i, cols-1, cols)] = 0.0f;     // right
    }
    // Top boundary (excluding corners) = 1
    for (int j = 1; j < cols - 1; ++j)
        grid[IDX(0, j, cols)] = 1.0f;
}

// --------------- Correctness check --------------------------
// Returns true if every cell satisfies the relative tolerance.
// Also checks finite values and unchanged boundaries.
// Prints a report when verbose=true.
inline bool check_correctness(const float* ref,      // CPU reference
                               const float* test,     // GPU result
                               int rows, int cols,
                               float* out_max_err = nullptr,
                               bool   verbose     = false)
{
    float max_err = 0.0f;
    bool  ok      = true;

    for (int i = 0; i < rows * cols; ++i) {
        if (!std::isfinite(test[i])) {
            if (verbose) std::printf("  Non-finite value at flat index %d\n", i);
            ok = false;
            continue;
        }
        float err = std::fabs(test[i] - ref[i]);
        float tol = REL_TOL * (1.0f + std::fabs(ref[i]));
        if (err > tol) ok = false;
        if (err > max_err) max_err = err;
    }

    // Boundary checks on the test grid
    auto boundary_ok = [&](int i, int j, float expected) {
        float v = test[IDX(i, j, cols)];
        if (std::fabs(v - expected) > 1e-7f) {
            if (verbose)
                std::printf("  Boundary violation at (%d,%d): got %.6f expected %.6f\n",
                            i, j, v, expected);
            ok = false;
        }
    };
    for (int j = 1; j < cols - 1; ++j) boundary_ok(0,      j, 1.0f);
    for (int j = 0; j < cols;     ++j) boundary_ok(rows-1, j, 0.0f);
    for (int i = 0; i < rows;     ++i) boundary_ok(i, 0,      0.0f);
    for (int i = 0; i < rows;     ++i) boundary_ok(i, cols-1, 0.0f);
    // corners
    boundary_ok(0,      0,      0.0f);
    boundary_ok(0,      cols-1, 0.0f);
    boundary_ok(rows-1, 0,      0.0f);
    boundary_ok(rows-1, cols-1, 0.0f);

    if (out_max_err) *out_max_err = max_err;
    return ok;
}

// --------------- Grid I/O helpers ---------------------------
// Save grid to raw binary file (rows*cols floats, little-endian).
inline void save_grid(const char* path, const float* grid, int rows, int cols)
{
    FILE* f = std::fopen(path, "wb");
    if (!f) throw std::runtime_error(std::string("Cannot open for write: ") + path);
    std::fwrite(grid, sizeof(float), (size_t)rows * cols, f);
    std::fclose(f);
}

// Load grid from raw binary file into pre-allocated buffer.
inline void load_grid(const char* path, float* grid, int rows, int cols)
{
    FILE* f = std::fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("Cannot open for read: ") + path);
    size_t n = std::fread(grid, sizeof(float), (size_t)rows * cols, f);
    std::fclose(f);
    if ((int)n != rows * cols)
        throw std::runtime_error("File size mismatch when loading grid");
}
