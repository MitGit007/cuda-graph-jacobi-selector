# Experimental results

The files in this directory are the final RTX 3050 records used for the project
evaluation.

| File | Contents |
| --- | --- |
| `heldout_raw.csv` | 690 timed runs: ordinary CUDA, selector and valid fixed batches |
| `heldout_order.csv` | Randomized execution order for each of five repetitions |
| `large_validation.txt` | Four graph and four selector comparisons against ordinary CUDA |

The held-out grid sides were 32, 1024 and 2048. Iteration counts were 33, 100,
511, 1000 and 5000. Candidate graph batches were 2, 4, 8, 16, 32, 64, 100 and
128, subject to `batch <= iterations`.

## Calibrated selector cases

Times are medians of five runs in milliseconds. The oracle is the fixed batch
with the lowest measured median total time for the same grid and iteration
count.

| Grid | Iterations | Ordinary | Selector | Selector choice | Speedup | Oracle batch | Oracle time |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 32 x 32 | 511 | 5.203 | 3.434 | 16 | 1.52x | 16 | 2.293 |
| 32 x 32 | 1000 | 10.422 | 4.227 | 16 in 4/5 runs; 128 in 1/5 | 2.47x | 16 | 2.638 |
| 32 x 32 | 5000 | 42.887 | 8.095 | 128 | 5.30x | 16 | 5.974 |
| 1024 x 1024 | 5000 | 282.740 | 286.054 | 128 | 0.988x | 16 | 272.983 |
| 2048 x 2048 | 5000 | 1081.316 | 1120.516 | 128 | 0.965x | 16 | 1070.447 |

The selector bypassed calibration and executed ordinary CUDA in the remaining
ten held-out configurations according to its frozen bypass rules.

## Correctness evidence

- All 230 CPU-referenced timed runs passed.
- Four large-grid graph outputs matched ordinary CUDA.
- Four large-grid selector outputs matched ordinary CUDA.
- The reported maximum absolute and relative errors were zero for these runs.
