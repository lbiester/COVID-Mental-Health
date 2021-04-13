"""
Script to generate Tables 4, 5, and 6
"""
import argparse
import itertools
import os
from typing import List, Optional

import pandas as pd
from argparse_utils import enum_action

from src.data_utils import get_mh_subreddits
from src.enums import PostType
from src.graph.process_and_aggregate_graph_features import SELECT_FEATURES_GRAPH
from src.liwc.liwc_processor import LIWCProcessor

PROPHET_OUTPUT_DIR = "data/prophet_output/"
TABLE_SETTINGS = "data/graph_settings.csv"
ALPHA = 0.05
BONFERRONI = 588
PROPHET_SETTING_STR = "2017-01-01.2020-03-01.2020-05-31.True.0.95.additive"
LIWC_CATEGORIES = LIWCProcessor().liwc_categories


# create dataframe to store results
def get_outside_ci_percent(categories: List[str], feature_suffix: str, post_type: Optional[PostType]) -> pd.DataFrame:
    """
    Create a dataframe to store all results
    :return: the dataframe
    """
    lower_subreddits = [subr.lower() for subr in get_mh_subreddits()]
    index_array = list(itertools.product(lower_subreddits, categories))
    index = pd.MultiIndex.from_tuples(index_array, names=("subreddit", "category"))
    outside_ci_percent = pd.DataFrame(dtype="object", index=index, columns=["outside_ci", "direction", "significant"])

    for subreddit in lower_subreddits:
        for category in categories:
            pt = f"_{post_type.name.lower()}" if post_type is not None else ""
            feature_suffix = f"_{feature_suffix}" if feature_suffix is not None else ""
            feature_path = f"{subreddit.lower()}{pt}{feature_suffix}_" \
                           f"baseline_avg{pt}{feature_suffix}_difference.{category}"
            setting_dir = os.path.join(PROPHET_OUTPUT_DIR, feature_path, PROPHET_SETTING_STR)
            setting_results = pd.read_csv(os.path.join(setting_dir, "computed_results.csv"), index_col=0)

            outside_ci = setting_results.loc["Outside CI post-COVID"].item()
            mean_diff = setting_results.loc["Mean difference outliers post-COVID"].item()
            pval = setting_results.loc["PVal Outside CI post-COVID"].item()

            outside_ci_percent.loc[subreddit, category]["outside_ci"] = outside_ci
            outside_ci_percent.loc[subreddit, category]["direction"] = "up" if mean_diff > 0 else "down"
            outside_ci_percent.loc[subreddit, category]["significant"] = pval * BONFERRONI < ALPHA
    return outside_ci_percent


def get_top_ten_by_subreddit(outside_ci_percent: pd.DataFrame, subreddit: str) -> pd.DataFrame:
    """
    Get the top 10 categories/topics with the largest number of outliers
    :param outside_ci_percent: The dataframe from get_outside_ci_percent
    :param subreddit: The subreddit to get the top 10 for
    :return: dataframe representing the largest outliers
    """
    return outside_ci_percent.loc[subreddit, :].sort_values(by="outside_ci", ascending=False)[:10].reset_index()


def add_subreddit_header(df: pd.DataFrame, subreddit: str) -> pd.DataFrame:
    """
    Add a new header to a dataframe which will show which subreddit the dataframe comes from
    :param df: the dataframe to add the header to
    :param subreddit: the subreddit name to use as the header
    :return: the dataframe with the new header added
    """
    df = df.copy()
    header = pd.MultiIndex.from_product([[subreddit], df.columns])
    df.columns = header
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type",
                        choices=["liwc", "topic", "graph"],
                        required=True,
                        help="The type of features to create tables for")
    parser.add_argument("--feature_suffix",
                        help="Additional suffix used in output - we use topic_dist for topics")
    parser.add_argument("-pt", "--post_type", dest="post_type", action=enum_action(PostType), required=False,
                        help="Post type to process - do not use for graph")
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.feature_type == "liwc":
        categories = LIWC_CATEGORIES
    elif args.feature_type == "graph":
        categories = SELECT_FEATURES_GRAPH
    else:
        categories = [f"topic_{i}" for i in range(25)]

    outside_ci_percent = get_outside_ci_percent(categories, args.feature_suffix, args.post_type)

    anxiety = add_subreddit_header(get_top_ten_by_subreddit(outside_ci_percent, "anxiety"), "Anxiety")
    depression = add_subreddit_header(get_top_ten_by_subreddit(outside_ci_percent, "depression"), "depression")
    suicidewatch = add_subreddit_header(get_top_ten_by_subreddit(outside_ci_percent, "suicidewatch"), "SuicideWatch")

    concatenated = pd.concat((anxiety, depression, suicidewatch), axis=1)
    print(concatenated)


if __name__ == "__main__":
    main()
