"""
Script to generate Figure 2
"""
import argparse
import os
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from argparse_utils import enum_action

from src.covid_lexicon.covid_lexicon_processor import CovidLexiconProcessor
from src.data_utils import get_mh_subreddits, get_subreddit_json
from src.enums import PostType
from src.liwc.compute_avg_liwc_features_by_dt import convert_timestamp


def plot_covid19_discussion(features: List[Tuple[str, datetime, np.array]], lexicon_words: List[str],
                            post_type: PostType, show_title: bool = False, large_figures: bool = False):
    post_type_str = post_type.name.lower()
    post_times = [(subr, utc) for subr, utc, _ in features]
    post_features = [feat for _, _, feat in features]
    df_1 = pd.DataFrame(post_times, columns=["subreddit", "created_utc"])
    df_2 = pd.DataFrame(post_features, columns=lexicon_words)
    df = pd.concat((df_1, df_2), axis=1)

    # 7 day rolling mean: how many posts/comments mention COVID-19?
    fontsize = 16
    fig, ax = plt.subplots(figsize=((20, 8) if large_figures else (6, 1.5)))
    for subreddit in get_mh_subreddits():
        df[df["subreddit"] == subreddit].set_index("created_utc").sum(axis=1)[:"2020-05"].apply(lambda x: int(x > 0))\
            .resample("D").mean().rolling(7).mean().dropna().mul(100).plot.line(label=f"r/{subreddit}", ax=ax,
                                                                                legend=True, fontsize=fontsize)
    ax.set_xlabel("Date", fontsize=fontsize)
    ax.set_ylabel(f"% of {post_type_str}s", fontsize=fontsize)
    ax.legend()
    if show_title:
        ax.set_title(f"Volume of {post_type_str}s mentioning COVID-19 words by subreddit")
    os.makedirs("graphs/covid_lexicon", exist_ok=True)
    plt.savefig(f"graphs/covid_lexicon/covid_lexicon_{post_type_str}.png")
    plt.close()

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--post_type", dest="post_type", action=enum_action(PostType), required=True,
                        help="Post type to process")
    args = parser.parse_args()
    return args


def main():
    """
    Retrieve covd lexicon count features then create a graph
    """
    args = _parse_args()

    clp = CovidLexiconProcessor()

    # retrieve the count features
    features_post = []
    for subreddit in get_mh_subreddits():
        for post in get_subreddit_json(subreddit, args.post_type, properties=["id", "created_utc"],
                                       start_time=datetime(2019, 12, 25)):
            features, _ = clp.read_post_by_id(post["id"], False)
            features_post.append((subreddit, convert_timestamp(post["created_utc"]), features))

    plot_covid19_discussion(features_post, clp.lexicon_words, args.post_type, large_figures=True)


if __name__ == "__main__":
    main()
