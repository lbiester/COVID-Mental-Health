"""
Convert pickled dataframe with graph features to CSV with format as expected by time_series_model_prophet.py
"""
import argparse

import pandas as pd
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Path to file to convert.")
    args = parser.parse_args()
    return args


def convert_file(file_path: str):
    df = pd.read_pickle(file_path)
    # make index interval_id
    df.set_index('interval_id', inplace=True)
    # rename misnamed column
    df.rename({'mean_shortest_path_distr': 'mean_shortest_path', 'u_mean_shortest_path_distr': 'u_mean_shortest_path'},
              axis=1, inplace=True)
    # convert distribution features to have column for each percentile
    dist_cols = [col for col in df.columns if 'distr' in col or 'dist' in col]
    base_col_names = ["_".join(col.split("_")[:-1]) for col in dist_cols]
    percentiles = np.arange(0, 101, 10)
    for row_idx, row in df.iterrows():
        for col, col_base_name in zip(dist_cols, base_col_names):
            weights, bins = row[col]
            # only add bins (percentiles as features to dataframe)
            assert len(bins) == 11, "found item with bins != 11"
            for val, percentile in zip(bins, percentiles):
                df.loc[row_idx, "{}_percentile_{}".format(col_base_name, percentile)] = val
    # drop old dist columns
    df.drop(dist_cols, axis=1, inplace=True)
    # also drop interval start and interval end columns
    df.drop(["interval_start", "interval_end"], axis=1, inplace=True)
    file_base_name = file_path.split(".")[0]
    df.to_csv("{}.csv".format(file_base_name))


def remove_duplicates(file_path: str):
    # because of error in grouping of graph computation features in user_graph.py (now fixed)
    # the features for some dates were computed multiple times --> need to remove duplicates
    df = pd.read_csv(file_path, index_col=0)
    df['temp_col'] = df.index
    df.drop_duplicates(inplace=True)
    df.drop('temp_col', axis=1, inplace=True)
    df.to_csv(file_path)


def main():
    args = _parse_args()
    convert_file(args.file_path)


if __name__ == "__main__":
    main()
