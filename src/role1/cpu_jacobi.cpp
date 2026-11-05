// ============================================================
// cpu_jacobi.cpp  —  CPU reference Jacobi solver
// ============================================================

#include "cpu_jacobi.h"
#include "../common.h"

#include <cstring>
#include <stdexcept>

void cpu_jacobi(const float* init, float* out, int rows, int cols, int N)
{
    if (N < 0) throw std::invalid_argument("N must be >= 0");
    if (!init || !out) throw std::invalid_argument("Null grid pointer");

    const int total = rows * cols;

    // Allocate two working buffers on the heap.
    // A is the "read" buffer, B is the "write" buffer.
    float* A = new float[total];
    float* B = new float[total];

    // Initialise both from the provided starting grid.
    std::memcpy(A, init, total * sizeof(float));
    std::memcpy(B, init, total * sizeof(float));

    float* src = A;
    float* dst = B;

    for (int iter = 0; iter < N; ++iter)
    {
        // Update every interior cell: i in [1, rows-2], j in [1, cols-2].
        // Reads only from src; writes only to dst.
        // Boundary cells in dst are never touched — they already hold
        // the correct boundary values (copied from init).
        for (int i = 1; i < rows - 1; ++i)
        {
            for (int j = 1; j < cols - 1; ++j)
            {
                dst[IDX(i, j, cols)] =
                    0.25f * (src[IDX(i-1, j, cols)] +
                             src[IDX(i+1, j, cols)] +
                             src[IDX(i, j-1, cols)] +
                             src[IDX(i, j+1, cols)]);
            }
        }

        // Swap roles: what was dst becomes the new src.
        float* tmp = src;
        src = dst;
        dst = tmp;
    }

    // After N iterations:
    //   N even → final data is in A  (src == A after even swaps)
    //   N odd  → final data is in B  (src == B after odd swaps)
    // Either way, src points to the answer.
    std::memcpy(out, src, total * sizeof(float));

    delete[] A;
    delete[] B;
}
