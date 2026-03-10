"""Print MLIR IR for the dynamic multicore rowsum or colsum kernel (fp32).

Usage: python gen_ir.py --mode row|col
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sum_builder import build_colsum, build_rowsum

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["row", "col"], default="row")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    args = parser.parse_args()
    builder = build_rowsum if args.mode == "row" else build_colsum
    print(builder(dtype=args.dtype))
