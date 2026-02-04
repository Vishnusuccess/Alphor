#!/usr/bin/env python3
"""Utility: split a CSV into stratified subsets of specified sizes.

Usage examples:
  python split_csv_stratified.py --input base_trainset.csv --target churn --sizes 2000 3000 --output_prefix client_split
  python split_csv_stratified.py --input data.csv --target label --sizes 2k 3k 5k --output_prefix parts

The script attempts to stratify by `--target`. If strict stratified sampling fails
for a requested size (too small to preserve all classes), it falls back to
random sampling for that part with a warning.
"""
from __future__ import annotations

import argparse
import math
from typing import List
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def parse_size_token(s: str) -> int:
    s = s.strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1000)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1_000_000)
    return int(float(s))


def split_csv_stratified(
    input_path: str,
    target: str,
    sizes: List[int],
    output_prefix: str,
    random_state: int = 42,
):
    df = pd.read_csv(input_path)
    n_total = len(df)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {input_path}")

    remaining = df.copy()
    outputs = []
    for i, size in enumerate(sizes, start=1):
        if size <= 0:
            continue
        if size >= len(remaining):
            subset = remaining.copy()
            remaining = remaining.iloc[0:0]
        else:
            # try stratified split on the remaining set
            try:
                frac = size / len(remaining)
                sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=random_state + i)
                y = remaining[target].values
                train_idx, test_idx = next(sss.split(remaining, y))
                subset = remaining.iloc[test_idx].copy()
                remaining = remaining.iloc[train_idx].copy()
            except Exception as e:
                # fallback: random sample without stratify
                print(f"Warning: stratified split failed for size={size} ({e}), doing random sample")
                subset = remaining.sample(n=size, random_state=random_state + i)
                remaining = remaining.drop(subset.index)

        out_name = f"{output_prefix}_part{i}_{len(subset)}.csv"
        subset.to_csv(out_name, index=False)
        outputs.append(out_name)
        print(f"Wrote {out_name} ({len(subset)} rows)")
        if remaining.empty:
            break

    if len(remaining) > 0:
        rem_name = f"{output_prefix}_remainder_{len(remaining)}.csv"
        remaining.to_csv(rem_name, index=False)
        outputs.append(rem_name)
        print(f"Wrote remainder {rem_name} ({len(remaining)} rows)")

    return outputs


def main():
    p = argparse.ArgumentParser(description="Stratified CSV splitter")
    p.add_argument("--input", "-i", required=True, help="Input CSV file")
    p.add_argument("--target", "-t", required=True, help="Target column for stratification")
    p.add_argument("--sizes", "-s", required=True, nargs="+", help="Requested subset sizes (e.g. 2000 3000 or 2k 3k)")
    p.add_argument("--output_prefix", "-o", default="split", help="Output filename prefix")
    p.add_argument("--random_state", type=int, default=42)

    args = p.parse_args()

    sizes = [parse_size_token(x) for x in args.sizes]
    total_requested = sum(sizes)
    if total_requested > 10_000_000 and total_requested > len(pd.read_csv(args.input)):
        print("Warning: requested more rows than present in the input")

    split_csv_stratified(args.input, args.target, sizes, args.output_prefix, args.random_state)


if __name__ == "__main__":
    main()
