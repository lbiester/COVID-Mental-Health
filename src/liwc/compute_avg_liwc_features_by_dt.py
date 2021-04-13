
import argparse
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from argparse_utils import enum_action
from pandas import DatetimeIndex
from tqdm import tqdm

from src.data_utils import get_subreddit_json, get_text
from src.enums import PostType
from src.liwc.liwc_reddit_processor import LIWCRedditProcessor

DATA_PATH = "data/timeseries/journal/raw/liwc"


def convert_timestamp(timestamp: int) -> datetime:
    datetime_obj = datetime.utcfromtimestamp(timestamp)
    return datetime(datetime_obj.year, datetime_obj.month, datetime_obj.day)


def _get_liwc_info_by_dt(subreddit: str, post_type: PostType, liwc_reddit_processor: LIWCRedditProcessor) \
        -> Tuple[Dict[datetime, np.array], Dict[datetime, int]]:
    properties = ["id", "author", "subreddit", "created_utc", "title", "selftext"] if post_type == PostType.POST \
        else ["id", "author", "subreddit", "created_utc", "body"]
    assert post_type != PostType.ALL
    is_comment = post_type == PostType.COMMENT
    # ignore automoderator for comments (only)
    subr_data = get_subreddit_json(subreddit, post_type, ignore_automoderator=is_comment, properties=properties)

    liwc_sums_by_dt = defaultdict(lambda: np.zeros(73, dtype="object"))
    liwc_counts_by_dt = Counter()
    for post in subr_data:
        post_text = get_text(post, include_title=True)
        features, word_count = liwc_reddit_processor.process_post(post["id"], is_comment, post_text, post["author"],
                                                                  post["subreddit"], post["created_utc"])
        if word_count > 0:
            post_date = convert_timestamp(post["created_utc"])
            liwc_sums_by_dt[post_date] += features
            liwc_counts_by_dt[post_date] += 1
    return liwc_sums_by_dt, liwc_counts_by_dt


def _compute_liwc_subreddit_csv_helper(subreddit: str, identifier: str,
                                       liwc_sums_by_dt: Dict[datetime, np.array],
                                       liwc_counts_by_dt: Dict[datetime, int],
                                       csv_columns: List[str]):
    file_path = f"{DATA_PATH}/{subreddit}_{identifier}.csv"

    subreddit_liwc_avg_df = pd.DataFrame(
        0, index=DatetimeIndex(pd.to_datetime(sorted(liwc_sums_by_dt.keys()))),
        columns=csv_columns)
    for dt, liwc_sums in liwc_sums_by_dt.items():
        n_datapoints = liwc_counts_by_dt[dt]
        document_liwc_avg = np.divide(liwc_sums, n_datapoints)
        data = np.concatenate((document_liwc_avg, [n_datapoints]))
        subreddit_liwc_avg_df.loc[dt] = data
    subreddit_liwc_avg_df.to_csv(file_path)


def compute_liwc_subreddit_csv(subreddit: str, post_type: PostType):
    # will create csv file containing avg LIWC features by day for posts, comments, posts + comments
    liwc_reddit_processor = LIWCRedditProcessor(include_title=True)

    # fetch post dictionaries
    post_liwc_sums_by_dt, post_liwc_counts_by_dt = _get_liwc_info_by_dt(subreddit, post_type, liwc_reddit_processor)

    # create CSV file with averages by day
    csv_columns = liwc_reddit_processor.liwc_processor.liwc_categories + ["n_datapoints"]
    _compute_liwc_subreddit_csv_helper(
        subreddit, post_type.name.lower(), post_liwc_sums_by_dt, post_liwc_counts_by_dt, csv_columns)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddits",
                        nargs="+",
                        required=True,
                        help="Subreddit(s) to process")
    parser.add_argument("--post_type", action=enum_action(PostType), default=PostType.POST, help="Post type to process")
    return parser.parse_args()


def main():
    args = _parse_args()
    for subreddit in tqdm(args.subreddits):
        compute_liwc_subreddit_csv(subreddit.lower(), args.post_type)


if __name__ == "__main__":
    main()
