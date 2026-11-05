#!/usr/bin/env python3
"""Summarize the held-out CUDA Graph experiment CSV."""

import argparse
import csv
import statistics
from collections import Counter, defaultdict


def load_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        for name in ("rows", "cols", "iterations", "seed", "batch_size",
                     "correctness_status"):
            row[name] = int(row[name])
        for name in ("calibration_ms", "setup_ms", "execution_ms", "total_ms"):
            row[name] = float(row[name])
    return rows


def median_total(rows):
    return statistics.median(row["total_ms"] for row in rows)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize medians, selector choices and fixed-batch oracle")
    parser.add_argument("csv_file", help="held-out raw timing CSV")
    args = parser.parse_args()

    rows = load_rows(args.csv_file)
    groups = defaultdict(list)
    for row in rows:
        key = (row["rows"], row["cols"], row["iterations"],
               row["mode"], row["batch_size"])
        groups[key].append(row)

    checked = [row for row in rows if row["correctness_status"] >= 0]
    failures = [row for row in checked if row["correctness_status"] == 0]

    workloads = sorted({(row["rows"], row["cols"], row["iterations"])
                        for row in rows})

    print(f"Runs: {len(rows)}")
    print(f"CPU-referenced runs: {len(checked)}; failures: {len(failures)}")
    print()
    print("Calibrated selector cases (median total time, ms)")
    print("grid,iters,ordinary,selector,choices,speedup,oracle_batch,oracle")

    for grid_rows, grid_cols, iterations in workloads:
        selector_rows = [row for row in rows
                         if row["rows"] == grid_rows
                         and row["cols"] == grid_cols
                         and row["iterations"] == iterations
                         and row["mode"] == "selector"]
        if not selector_rows or not any(row["calibration_ms"] > 0.0
                                        for row in selector_rows):
            continue

        ordinary_rows = groups[(grid_rows, grid_cols, iterations, "ordinary", 0)]
        graph_groups = [
            (batch, values)
            for (r, c, n, mode, batch), values in groups.items()
            if (r, c, n) == (grid_rows, grid_cols, iterations)
            and mode == "graph"
        ]

        ordinary = median_total(ordinary_rows)
        selector = median_total(selector_rows)
        oracle_batch, oracle_rows = min(graph_groups,
                                        key=lambda item: median_total(item[1]))
        oracle = median_total(oracle_rows)
        choices = Counter(row["batch_size"] for row in selector_rows)
        choice_text = ";".join(f"{batch}:{count}"
                               for batch, count in sorted(choices.items()))
        speedup = ordinary / selector

        print(f"{grid_rows}x{grid_cols},{iterations},{ordinary:.3f},"
              f"{selector:.3f},{choice_text},{speedup:.3f},"
              f"{oracle_batch},{oracle:.3f}")


if __name__ == "__main__":
    main()
