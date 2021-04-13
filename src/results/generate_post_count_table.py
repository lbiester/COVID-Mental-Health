"""
Script to generate Table 1
"""
import os
from collections import Counter
from datetime import datetime, timezone
from statistics import mean

import pandas as pd

from src.data_utils import get_subreddit_json, get_baseline_subreddits, get_text
from src.enums import PostType
from src.text_utils import remove_special_chars

CHAR_COUNT_DIR = "data/timeseries/journal/raw/char_count"


def _read_counts(subreddit: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join("data/timeseries/journal/raw/post_count", f"{subreddit.lower()}.csv"),
                     index_col=[0], parse_dates=[0])
    return df


def _mean_counts_baseline() -> float:
    all_counts = []
    for subreddit in get_baseline_subreddits():
        all_counts.append(_read_counts(subreddit))
    return pd.concat(all_counts, axis=0).resample("D").mean()


def _get_yearly_mean(count_df: pd.DataFrame, post_type: PostType, year: int) -> int:
    if post_type == PostType.POST:
        return int(round(count_df[str(year)].mean()["post_count"], 0))
    else:
        return int(round(count_df[str(year)].mean()["comment_count"], 0))


def _get_daily_weighted_mean_char_count(subreddit: str, post_type: PostType) -> int:
    # try reading data from disk if it has already been saved
    csv_path = os.path.join(CHAR_COUNT_DIR, f"{subreddit}_{post_type.name.lower()}.csv")
    if os.path.isfile(csv_path):
        return int(round(pd.read_csv(csv_path, index_col=0, squeeze=True).mean()))

    daily_sums = Counter()
    daily_counts = Counter()
    for post in get_subreddit_json(subreddit, post_type, properties=["created_utc", "title", "selftext", "body"],
                                   start_time=datetime(2017, 1, 1), end_time=datetime(2020, 6, 1),
                                   ignore_automoderator=(post_type == PostType.COMMENT)):
        length = len(remove_special_chars(get_text(post, include_title=True)))
        dt = datetime.fromtimestamp(post["created_utc"], timezone.utc).date()
        daily_sums[dt] += length
        daily_counts[dt] += 1
    daily_avgs = pd.Series({dt: daily_sums[dt] / daily_counts[dt] for dt in daily_counts.keys()})

    # save data to disk (this takes forever to do for some subreddits)
    if not os.path.exists(CHAR_COUNT_DIR):
        os.mkdir(CHAR_COUNT_DIR)
    daily_avgs.to_csv(csv_path)
    return int(round(daily_avgs.mean()))


def _get_baseline_daily_weighted_mean_char_count(post_type: PostType) -> int:
    averages = []
    for subreddit in get_baseline_subreddits():
        averages.append(_get_daily_weighted_mean_char_count(subreddit, post_type))
    return int(round(mean(averages)))


def main():
    """
    For all subreddits, get avg yearly post/comment counts for 2017-2020, along with avg characters per document
    and display them
    """

    # read post/comment counts from disk
    anxiety_counts = _read_counts("anxiety")
    depression_counts = _read_counts("depression")
    suicidewatch_counts = _read_counts("suicidewatch")
    baseline_counts = _mean_counts_baseline()

    # create table of means + add char counts
    data_list = [["", "r/Anxiety", "", "r/depression", "", "r/SuicideWatch", "", "Control", ""],
                 ["", "P", "C", "P", "C", "P", "C", "P", "C"],
                 ["2017",
                  _get_yearly_mean(anxiety_counts, PostType.POST, 2017),
                  _get_yearly_mean(anxiety_counts, PostType.COMMENT, 2017),
                  _get_yearly_mean(depression_counts, PostType.POST, 2017),
                  _get_yearly_mean(depression_counts, PostType.COMMENT, 2017),
                  _get_yearly_mean(suicidewatch_counts, PostType.POST, 2017),
                  _get_yearly_mean(suicidewatch_counts, PostType.COMMENT, 2017),
                  _get_yearly_mean(baseline_counts, PostType.POST, 2017),
                  _get_yearly_mean(baseline_counts, PostType.COMMENT, 2017)],
                 ["2018",
                  _get_yearly_mean(anxiety_counts, PostType.POST, 2018),
                  _get_yearly_mean(anxiety_counts, PostType.COMMENT, 2018),
                  _get_yearly_mean(depression_counts, PostType.POST, 2018),
                  _get_yearly_mean(depression_counts, PostType.COMMENT, 2018),
                  _get_yearly_mean(suicidewatch_counts, PostType.POST, 2018),
                  _get_yearly_mean(suicidewatch_counts, PostType.COMMENT, 2018),
                  _get_yearly_mean(baseline_counts, PostType.POST, 2018),
                  _get_yearly_mean(baseline_counts, PostType.COMMENT, 2018)],
                 ["2019",
                  _get_yearly_mean(anxiety_counts, PostType.POST, 2018),
                  _get_yearly_mean(anxiety_counts, PostType.COMMENT, 2018),
                  _get_yearly_mean(depression_counts, PostType.POST, 2018),
                  _get_yearly_mean(depression_counts, PostType.COMMENT, 2018),
                  _get_yearly_mean(suicidewatch_counts, PostType.POST, 2018),
                  _get_yearly_mean(suicidewatch_counts, PostType.COMMENT, 2018),
                  _get_yearly_mean(baseline_counts, PostType.POST, 2018),
                  _get_yearly_mean(baseline_counts, PostType.COMMENT, 2018)],
                 ["2020",
                  _get_yearly_mean(anxiety_counts, PostType.POST, 2020),
                  _get_yearly_mean(anxiety_counts, PostType.COMMENT, 2020),
                  _get_yearly_mean(depression_counts, PostType.POST, 2020),
                  _get_yearly_mean(depression_counts, PostType.COMMENT, 2020),
                  _get_yearly_mean(suicidewatch_counts, PostType.POST, 2020),
                  _get_yearly_mean(suicidewatch_counts, PostType.COMMENT, 2020),
                  _get_yearly_mean(baseline_counts, PostType.POST, 2020),
                  _get_yearly_mean(baseline_counts, PostType.COMMENT, 2020)],
                 ["Chars",
                  _get_daily_weighted_mean_char_count("anxiety", PostType.POST),
                  _get_daily_weighted_mean_char_count("anxiety", PostType.COMMENT),
                  _get_daily_weighted_mean_char_count("depression", PostType.POST),
                  _get_daily_weighted_mean_char_count("depression", PostType.COMMENT),
                  _get_daily_weighted_mean_char_count("suicidewatch", PostType.POST),
                  _get_daily_weighted_mean_char_count("suicidewatch", PostType.COMMENT),
                  _get_baseline_daily_weighted_mean_char_count(PostType.POST),
                  _get_baseline_daily_weighted_mean_char_count(PostType.COMMENT)]
                 ]

    # display as a "table"
    for item in data_list:
        for thing in item:
            print(f"{thing}\t", end="")
        print()


if __name__ == "__main__":
    main()
