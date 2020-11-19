"""
Script to generate Figures 2, 3, and 4
"""
import argparse
import itertools
import os
from datetime import datetime
from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from tqdm import tqdm

from src.data_utils import get_mh_subreddits

PROPHET_OUTPUT_DIR = "data/prophet_output/"
GRAPH_SETTINGS = "data/graph_settings.csv"
ALPHA = 0.05
PROPHET_SETTING_STR = "2017-01-01.2020-03-01.2020-05-31.True.0.95.additive"


def plot_each_subreddit_plus_individual(feature_type: str, feature_name: str,
                                        ylabel: str = "% of words", multiplier: int = 100,
                                        title_category: Optional[str] = None, show_title: bool = False,
                                        setting: str = PROPHET_SETTING_STR,
                                        post_type: Optional[str] = None, include_legend: bool = False,
                                        bonferroni_num: Optional[int] = None, large_fonts: bool = True,
                                        suffix: Optional[str] = None):
    """
    series_category: this can just be whatever your series is called in prophet. it's also used as the title by
        default, but doesn't have to (can pass in title_category)
    ylabel: defaults to "% of words" but can be changed
    multiplier: defaults to 100, assuming you want a percent but your series is not multiplied yet. Set to 1 if it is
        already multiplied by 100!
    title_category: optional category name to use in the title if you don't want to use
    """
    title_fonts = 17 if large_fonts else None
    other_fonts = 16 if large_fonts else None

    # create dataframe to store results
    index_array = list(itertools.product(get_mh_subreddits(),
                                         pd.date_range(datetime(2017, 1, 1), datetime(2020, 5, 31))))
    index = pd.MultiIndex.from_tuples(index_array, names=("subreddit", "date"))
    timeseries = pd.DataFrame(0.0, index=index, columns=["True", "Forecast", "Upper", "Lower"])

    pt = f"_{post_type}" if post_type is not None else ""
    suffix = f"_{suffix}" if suffix is not None and len(suffix) > 0 else ""

    is_significant = {}
    for subreddit in get_mh_subreddits():
        setting_dir = os.path.join(
            PROPHET_OUTPUT_DIR, f"{subreddit.lower()}{pt}{suffix}.{feature_name}",
            setting)
        preprocessed_ts = pd.read_csv(os.path.join(setting_dir, "preprocessed_timeseries.csv"), index_col=0,
                                      squeeze=True, parse_dates=[0])
        forecast = pd.read_csv(os.path.join(setting_dir, "forecast.csv"), index_col=0, parse_dates=[0],
                               usecols=["ds", "yhat", "yhat_lower", "yhat_upper"])
        timeseries.loc[pd.IndexSlice[subreddit, :], "True"] = preprocessed_ts.values
        timeseries.loc[pd.IndexSlice[subreddit, :], "Forecast"] = forecast["yhat"].values
        timeseries.loc[pd.IndexSlice[subreddit, :], "Upper"] = forecast["yhat_upper"].values
        timeseries.loc[pd.IndexSlice[subreddit, :], "Lower"] = forecast["yhat_lower"].values

        computed_results = pd.read_csv(os.path.join(setting_dir, "computed_results.csv"), squeeze=True, index_col=0)
        pval = computed_results["PVal Outside CI post-COVID"]
        outside_ci = computed_results["Outside CI post-COVID"]
        cutoff = 0.05 / bonferroni_num
        is_significant[subreddit] = pval < cutoff and outside_ci >= 5

    grey_map = cm.get_cmap("Greys")
    blue_map = cm.get_cmap("Blues")

    # create plot starting in 2019
    fig, axes = plt.subplots(figsize=(6, 5), nrows=3, sharex=True)
    start = "2019"
    legend = False
    for i, subreddit in enumerate(get_mh_subreddits()):

        ax = axes.flat[i]
        sub_series = timeseries.loc[subreddit][start:].mul(multiplier)
        sub_series["True"].plot.line(ax=ax, label="true values", legend=legend,
                                     color=grey_map(0.8), fontsize=other_fonts)
        sub_series["Forecast"].plot.line(ax=ax, label="forecast", legend=legend,
                                         color=grey_map(0.5), fontsize=other_fonts)
        ax.fill_between(sub_series.index, sub_series["Lower"], sub_series["Upper"], color=blue_map(0.3),
                        label="95% prediction interval (shaded)")
        add = "*" if is_significant[subreddit] else ""
        ax.set_title(f"r/{subreddit} Difference {add}", fontsize=title_fonts)
        ax.axvline(datetime(2020, 3, 1), color="grey", linestyle="dotted")
        ax.set_xlabel("Date", fontsize=other_fonts)
    handles, labels = axes[0].get_legend_handles_labels()

    handles[2] = Patch(facecolor=grey_map(0.3), edgecolor=grey_map(0.3), label='Color Patch')

    fig.text(0, 0.5, ylabel, va="center", rotation="vertical", fontsize=other_fonts)

    if show_title:
        title_category = title_category if title_category is not None else feature_name
        fig.suptitle(f"{title_category} Over Time")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        plt.tight_layout()

    os.makedirs(os.path.join("graphs", feature_type), exist_ok=True)
    plt.savefig(f"graphs/{feature_type}/{feature_name}{pt}.png")
    plt.close(fig)
    plt.close(fig)
    if include_legend:
        # legends are saved in external figures
        fig_legend = plt.figure(figsize=(9, .6))
        plt.figlegend(handles=handles, labels=labels, ncol=3, loc=2, prop={'size': other_fonts})
        fig_legend.savefig("legend_top.png")
        fig_legend = plt.figure(figsize=(9.2, .6))
        handles_2, labels_2 = axes[3].get_legend_handles_labels()
        plt.figlegend(handles=handles_2, labels=labels_2, ncol=4, loc=2, prop={'size': other_fonts})
        fig_legend.savefig("legend_bottom.png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type",
                        choices=["user_count", "liwc", "topic"],
                        required=True,
                        help="The type of features to create graphs for")
    return parser.parse_args()


def main():
    args = _parse_args()
    graph_settings_df = pd.read_csv(GRAPH_SETTINGS).fillna("")
    graph_settings = graph_settings_df[graph_settings_df["feature_type"] == args.feature_type].to_dict(orient="records")

    for setting in tqdm(graph_settings):
        bonferroni = len(graph_settings_df[graph_settings_df["bonferroni_group"] == setting["bonferroni_group"]]) * 3
        plot_each_subreddit_plus_individual(setting["feature_type"], setting["feature_name"], post_type="post",
                                            multiplier=1, ylabel=setting["Axis Label"], large_fonts=True,
                                            suffix=setting["suffix"], bonferroni_num=bonferroni)


if __name__ == "__main__":
    main()
