#!/usr/bin/env python3
"""
compare.py  —  Grid comparison helper script

Usage:
    python scripts/compare.py <file_a.bin> <file_b.bin> <rows> <cols> [--tol 1e-5]

Loads two raw float32 grid files and prints:
  - Max absolute difference
  - Max relative difference  (|a-b| / (1 + |a|))
  - Whether they agree within the project tolerance REL_TOL = 1e-5
  - A table of the 10 largest discrepancies

Exit code 0 = within tolerance, 1 = differences exceed tolerance.
"""

import sys
import argparse
import struct
import math

def load_grid(path, rows, cols):
    total = rows * cols
    with open(path, 'rb') as f:
        data = f.read(total * 4)  # 4 bytes per float32
    if len(data) != total * 4:
        raise ValueError(f"{path}: expected {total*4} bytes, got {len(data)}")
    return list(struct.unpack(f'{total}f', data))

def main():
    parser = argparse.ArgumentParser(description='Compare two Jacobi grid files')
    parser.add_argument('file_a', help='First grid (reference)')
    parser.add_argument('file_b', help='Second grid (test)')
    parser.add_argument('rows',   type=int)
    parser.add_argument('cols',   type=int)
    parser.add_argument('--tol',  type=float, default=1e-5,
                        help='Relative tolerance (default: 1e-5)')
    args = parser.parse_args()

    a = load_grid(args.file_a, args.rows, args.cols)
    b = load_grid(args.file_b, args.rows, args.cols)

    REL_TOL = args.tol
    total = args.rows * args.cols

    max_abs = 0.0
    max_rel = 0.0
    violations = []

    for idx, (va, vb) in enumerate(zip(a, b)):
        row = idx // args.cols
        col = idx  % args.cols
        if not math.isfinite(vb):
            violations.append((float('inf'), row, col, va, vb))
            continue
        abs_err = abs(va - vb)
        rel_err = abs_err / (1.0 + abs(va))
        if abs_err > max_abs:
            max_abs = abs_err
        if rel_err > max_rel:
            max_rel = rel_err
        if rel_err > REL_TOL:
            violations.append((rel_err, row, col, va, vb))

    print(f"Grid: {args.rows}x{args.cols}  ({total} cells)")
    print(f"Max absolute error : {max_abs:.6e}")
    print(f"Max relative error : {max_rel:.6e}  (tolerance: {REL_TOL:.1e})")
    print(f"Violations         : {len(violations)}")

    if violations:
        violations.sort(reverse=True)
        print(f"\nTop {min(10, len(violations))} discrepancies:")
        print(f"  {'row':>5} {'col':>5}  {'ref':>14} {'test':>14}  {'rel_err':>12}")
        for rel_err, row, col, va, vb in violations[:10]:
            print(f"  {row:5d} {col:5d}  {va:14.7f} {vb:14.7f}  {rel_err:12.3e}")
        print()
        print("Result: FAIL")
        return 1

    print("Result: PASS")
    return 0

if __name__ == '__main__':
    sys.exit(main())
