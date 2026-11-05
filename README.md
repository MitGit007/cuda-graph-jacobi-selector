# CUDA Graph Batch Selection for an Iterative 2D Jacobi Solver

This repository contains the implementation and experimental scripts for the
PCAP mini-project **A Lightweight Runtime Selector for CUDA Graph Batch Sizes
in an Iterative 2D Jacobi Solver**.

The program evaluates three execution policies that all use the same Jacobi
CUDA kernel:

1. ordinary CUDA kernel launches;
2. fixed-size CUDA Graph batches; and
3. a runtime selector that chooses ordinary execution or one of three measured
   graph batches.

The implementation uses fixed iteration counts and two ping-pong grid buffers.
Graph batches use even sizes so that each replay preserves buffer parity.
Remainder iterations are executed with ordinary launches.

## Authors

- Pratheek R. Rayakar — 240905636
- Sagar Riky Rodrigues — 240905660
- Shloka Shetty — 240905051

School of Computer Engineering, Manipal Institute of Technology, Manipal
Academy of Higher Education.

## Repository structure

```text
src/                 C++ and CUDA implementation
tests/               CPU/GPU correctness tests
scripts/             validation and benchmark automation
results/             final RTX 3050 experimental records
CMakeLists.txt        build configuration
```

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit 13.1 or a compatible toolkit
- CMake 3.18 or later
- C++17 compiler
- Visual Studio 2022 Build Tools on Windows
- Python 3 for large-grid output comparison
- PowerShell for the supplied experiment scripts

The reported experiments used an NVIDIA GeForce RTX 3050 Laptop GPU with
6 GiB VRAM, compute capability 8.6, CUDA Toolkit 13.1 and Windows 11.

## Build on Windows

Open the **x64 Native Tools Command Prompt for VS 2022**, change to the
repository directory, and run:

```cmd
cmake -S . -B build-3050 -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-3050
```

For a different NVIDIA architecture, replace `86` with the appropriate CMake
CUDA architecture value.

## Correctness tests

```cmd
build-3050\jacobi_tests.exe
```

The tests compare ordinary CUDA output with an independent CPU
implementation. They include square and rectangular grids, odd and even
iteration counts, fixed boundaries, a hand-checkable 5 x 5 update, and
ping-pong-buffer parity.

## Solver examples

Ordinary CUDA launches:

```cmd
build-3050\jacobi.exe --mode ordinary --rows 32 --cols 32 --iters 5000 --seed 42
```

Fixed CUDA Graph batch:

```cmd
build-3050\jacobi.exe --mode graph --rows 32 --cols 32 --iters 5000 --batch 16 --seed 42
```

Runtime selector:

```cmd
build-3050\jacobi.exe --mode selector --rows 32 --cols 32 --iters 5000 --seed 42
```

Available graph batch sizes are `2, 4, 8, 16, 32, 64, 100, 128`. A batch must
not exceed the requested iteration count.

## Reproduce the experiments

Large-grid correctness validation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\validate_large.ps1
```

Held-out performance sweep with five repetitions:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_heldout.ps1 -Repetitions 5
```

Summarize a completed raw timing file:

```cmd
python scripts\summarize_results.py results\heldout_raw.csv
```

The final sweep uses grid sides `32`, `1024`, and `2048`; iteration counts
`33`, `100`, `511`, `1000`, and `5000`; ordinary CUDA, the runtime selector,
and every valid fixed batch. Execution order is randomized independently for
each repetition.

## Timing and validation

End-to-end GPU time includes input transfers, selector calibration when used,
final graph construction, solver execution, synchronization, and output
transfer. CUDA context initialization, device allocation, stream creation, and
the CPU reference computation are outside the timed interval.

Grids up to 512 x 512 are compared directly with the CPU reference using

```text
abs(GPU - CPU) <= 1e-5 * (1 + abs(CPU)).
```

Larger grids are validated separately by comparing graph and selector outputs
with ordinary CUDA output. The supplied final validation log reports zero
tolerance violations in all eight large-grid comparisons.

## Main experimental finding

CUDA Graph batching substantially reduced end-to-end time for the smallest,
launch-bound workload. The runtime selector improved three calibrated 32 x 32
cases over ordinary launches, but its calibration cost was not recovered on the
1024 x 1024 and 2048 x 2048 cases. Fixed batch size 16 was the best measured
candidate in the five configurations where calibration ran.

See [results/README.md](results/README.md) for the final data files and summary.
