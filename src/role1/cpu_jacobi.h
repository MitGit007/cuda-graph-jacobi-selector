#pragma once
// ============================================================
// cpu_jacobi.h  —  CPU reference Jacobi solver
//
// Provides an independent, easy-to-verify implementation used
// as the correctness oracle for all GPU variants.
// ============================================================

// Run exactly N Jacobi iterations on a rows x cols grid.
//
// Parameters:
//   init   - initial grid (rows*cols floats, row-major)
//            Boundaries must already be set as per common.h.
//   out    - caller-allocated output buffer (rows*cols floats)
//   rows   - total rows INCLUDING boundary cells
//   cols   - total cols INCLUDING boundary cells
//   N      - exact number of iterations (do not stop early)
//
// After the call, 'out' contains the final grid.
void cpu_jacobi(const float* init, float* out, int rows, int cols, int N);
