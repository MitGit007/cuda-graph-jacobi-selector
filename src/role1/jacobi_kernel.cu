// ============================================================
// jacobi_kernel.cu  —  Device-side definition of jacobi_update
//
// Compiled ONCE into its own object.  All other .cu files that
// need to launch this kernel include jacobi_kernel.cuh for the
// declaration and launch_config helper only.
// ============================================================

#include "jacobi_kernel.cuh"

__global__ void jacobi_update(const float* __restrict__ src,
                                     float* __restrict__ dst,
                               int rows, int cols)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i >= rows - 1 || j >= cols - 1) return;

    dst[IDX(i, j, cols)] =
        0.25f * (src[IDX(i-1, j, cols)] +
                 src[IDX(i+1, j, cols)] +
                 src[IDX(i, j-1, cols)] +
                 src[IDX(i, j+1, cols)]);
}
