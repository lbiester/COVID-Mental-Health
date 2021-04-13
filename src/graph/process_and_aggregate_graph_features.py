import argparse
import os

import numpy as np
import pandas as pd


"""
Script for processing the features produced by user_graph.py

Used for aggregating features across the different output files and 
for reducing features considered to just those deemed most relevant.

Notes on feature selected as relevant (see SELECT_FEATURES_GRAPH below)
- Decided to only use undirected graph metrics. This was done to reduce redundancy with the 
  directed graph metrics and because the undirected graph is a bit simpler than the directed graph. Also, I think
  direction is less relevant to Reddit as activity is less about back and forth interactions.
- Using all non-percentile/distribution undirected graph features, except for eccentricity. It is excluded because it 
  is similar to diameter and mean_shortest_path, but I don't think it's used as often.
- Excluding all distribution/percentile features because there is so many that we risk finding spurious correlations.
"""


SELECT_FEATURES_GRAPH = [
    "u_connected_component_count",
    "u_mean_connected_component_size",
    "u_mean_shortest_path",
    "u_diameter",
    "u_node_count",
    "u_edge_count",
    "u_density",
    "u_mean_clustering_coefficient",
    "u_mean_degree",
    "u_assortativity",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, help="Path to directory containing feature files.")
    parser.add_argument("--output_file_name", type=str, help="Prefix to use in naming output csv file.")
    parser.add_argument("--file_name_fmt", type=str, help="Basic naming scheme for feature files.")
    parser.add_argument("--num_groups", type=int, help="Number of group files features were split into.")
    args = parser.parse_args()
    return args


def combine_files(feature_dir: str, file_name_fmt: str, num_groups: int) -> pd.DataFrame:
    dfs = []
    for group_num in range(num_groups):
        df = pd.read_pickle(os.path.join(feature_dir, file_name_fmt.format(group_num)))[SELECT_FEATURES_GRAPH]
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df


def process_features(df: pd.DataFrame):
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
    return df


def select_features(df, output_dir, output_file_name):
    ignore_feats = [col for col in df.columns if col not in set(SELECT_FEATURES_GRAPH)]
    df_select = df.drop(ignore_feats, axis=1)
    df_select.to_csv(os.path.join(output_dir, "{}.csv".format(output_file_name)))


def main():
    args = _parse_args()
    combined_df = combine_files(args.feature_dir, args.file_name_fmt, args.num_groups)
    processed_df = process_features(combined_df)
    select_features(processed_df, args.feature_dir, args.output_file_name)


if __name__ == "__main__":
    main()