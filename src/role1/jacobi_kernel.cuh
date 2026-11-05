#pragma once
// ============================================================
// jacobi_kernel.cuh  —  Shared CUDA kernel declaration
//
// The kernel body lives in jacobi_kernel.cu (compiled once).
// Every other .cu file includes this header for the declaration
// and the launch_config helper only — no duplicate definitions.
//
// THIS IS THE SINGLE KERNEL USED BY ALL GPU MODES:
//   • Ordinary launches
//   • CUDA Graph capture/replay
//   • Runtime selector
// ============================================================

#include "../common.h"
#include <cuda_runtime.h>

// Fixed block dimensions — 16x16 = 256 threads/block (suits sm_86).
static constexpr int BLOCK_DIM_X = 16;
static constexpr int BLOCK_DIM_Y = 16;

// ---- Kernel declaration ------------------------------------
// Definition is in jacobi_kernel.cu (compiled once).
// One GPU thread computes one interior cell of the Jacobi stencil.
//
//   src   - read-only input grid  (d_A or d_B)
//   dst   - write-only output grid (d_B or d_A)
//   rows  - total rows including boundaries
//   cols  - total cols including boundaries
//
// Boundary cells are never written by this kernel.
// Buffer alternation is handled by the caller (runner / graph capture).
__global__ void jacobi_update(const float* __restrict__ src,
                                     float* __restrict__ dst,
                               int rows, int cols);

// ---- Launch configuration helper --------------------------
// Computes block/grid dims that cover all interior cells.
// Call before every launch or graph capture.
//
// Usage:
//   dim3 blk, grd;
//   launch_config(rows, cols, blk, grd);
//   jacobi_update<<<grd, blk, 0, stream>>>(src, dst, rows, cols);
inline void launch_config(int rows, int cols, dim3& block, dim3& grid)
{
    block = dim3(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    int interior_cols = cols - 2;
    int interior_rows = rows - 2;
    grid  = dim3((interior_cols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                 (interior_rows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y,
                 1);
}
