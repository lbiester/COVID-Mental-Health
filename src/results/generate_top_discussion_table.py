"""
Script to generate Tables 2 and 3
"""
import argparse
import itertools
import os
from typing import List

import pandas as pd

from src.data_utils import get_mh_subreddits
from src.liwc.liwc_processor import LIWCProcessor

PROPHET_OUTPUT_DIR = "data/prophet_output/"
TABLE_SETTINGS = "data/graph_settings.csv"
ALPHA = 0.05
BONFERRONI = 294
PROPHET_SETTING_STR = "2017-01-01.2020-03-01.2020-05-31.True.0.95.additive"
LIWC_CATEGORIES = LIWCProcessor().liwc_categories


# create dataframe to store results
def get_outside_ci_percent(categories: List[str], feature_format_str: str) -> pd.DataFrame:
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
            if feature_format_str is None:
                feature_name = f"post.{category}"
            else:
                feature_name = feature_format_str.format("post", category)
            setting_dir = os.path.join(PROPHET_OUTPUT_DIR, f"{subreddit.lower()}_{feature_name}",
                                       PROPHET_SETTING_STR)
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
                        choices=["liwc", "topic"],
                        required=True,
                        help="The type of features to create graphs for")
    parser.add_argument("--feature_format_str",
                        help="Format str (using {}) that will be filled with the feature name and post type to load"
                             "results")
    # NOTE: for topics, with the file setup we have, feature_format_str should be 17_20_{}_topic_dist.{}
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.feature_type == "liwc":
        categories = LIWC_CATEGORIES
    else:
        categories = [f"topic_{i}" for i in range(25)]

    outside_ci_percent = get_outside_ci_percent(categories, args.feature_format_str)

    anxiety = add_subreddit_header(get_top_ten_by_subreddit(outside_ci_percent, "anxiety"), "Anxiety")
    depression = add_subreddit_header(get_top_ten_by_subreddit(outside_ci_percent, "depression"), "depression")
    suicidewatch = add_subreddit_header(get_top_ten_by_subreddit(outside_ci_percent, "suicidewatch"), "SuicideWatch")

    concatenated = pd.concat((anxiety, depression, suicidewatch), axis=1)
    print(concatenated)


if __name__ == "__main__":
    main()
