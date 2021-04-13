"""
Script to combine dataframes holding graph features for subsets of the dataset.
"""
import argparse
import os

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, help="Path to directory containing feature files.")
    parser.add_argument("--file_name_fmt", type=str, help="Basic naming scheme for feature files.")
    parser.add_argument("--num_groups", type=int, help="Number of group files features were split into.")
    args = parser.parse_args()
    return args


def combine_files(feature_dir: str, file_name_fmt: str, num_groups: int):
    dfs = []
    for group_num in range(num_groups):
        df = pd.read_pickle(os.path.join(feature_dir, file_name_fmt.format(group_num)))
        dfs.append(df)
    out_df = pd.concat(dfs)
    out_df.to_pickle(os.path.join(feature_dir, file_name_fmt.format("all")))


def main():
    args = _parse_args()
    combine_files(args.feature_dir, args.file_name_fmt, args.num_groups)


if __name__ == "__main__":
    main()
