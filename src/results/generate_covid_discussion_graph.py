"""
Script to generate Figure 1
"""
import os
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.covid_lexicon.covid_lexicon_processor import CovidLexiconProcessor
from src.data_utils import get_mh_subreddits, get_subreddit_json
from src.enums import PostType
from src.liwc.compute_avg_liwc_features_by_dt import convert_timestamp


def plot_covid19_discussion(features: List[Tuple[str, datetime, np.array]], lexicon_words: List[str],
                            show_title: bool = False, large_figures: bool = False):
    post_times = [(subr, utc) for subr, utc, _ in features]
    post_features = [feat for _, _, feat in features]
    df_1 = pd.DataFrame(post_times, columns=["subreddit", "created_utc"])
    df_2 = pd.DataFrame(post_features, columns=lexicon_words)
    df = pd.concat((df_1, df_2), axis=1)

    # 7 day rolling mean: how many posts mention COVID-19?
    fontsize = 16
    fig, ax = plt.subplots(figsize=((20, 8) if large_figures else (6, 1.5)))
    for subreddit in get_mh_subreddits():
        df[df["subreddit"] == subreddit].set_index("created_utc").sum(axis=1)[:"2020-05"].apply(lambda x: int(x > 0))\
            .resample("D").mean().rolling(7).mean().dropna().mul(100).plot.line(label=f"r/{subreddit}", ax=ax,
                                                                                legend=True, fontsize=fontsize)
    ax.set_xlabel("Date", fontsize=fontsize)
    ax.set_ylabel(f"% of posts", fontsize=fontsize)
    ax.legend()
    if show_title:
        ax.set_title(f"Volume of posts mentioning COVID-19 words by subreddit")
    os.makedirs("graphs/covid_lexicon", exist_ok=True)
    plt.savefig("graphs/covid_lexicon/covid_lexicon_post.png")
    plt.close()


def main():
    """
    Retrieve covd lexicon count features then create a graph
    """
    clp = CovidLexiconProcessor()

    # retrieve the count features
    features_post = []
    for subreddit in get_mh_subreddits():
        for post in get_subreddit_json(subreddit, PostType.POST, properties=["id", "created_utc"],
                                       start_time=datetime(2019, 12, 25)):
            features, _ = clp.read_post_by_id(post["id"], False)
            features_post.append((subreddit, convert_timestamp(post["created_utc"]), features))

    plot_covid19_discussion(features_post, clp.lexicon_words, large_figures=True)


if __name__ == "__main__":
    main()
