import argparse
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas import DatetimeIndex
from tqdm import tqdm

from src.data_utils import get_subreddit_json, get_text
from src.enums import PostType
from src.liwc.liwc_reddit_processor import LIWCRedditProcessor

DATA_PATH = "data/timeseries/liwc/"


def convert_timestamp(timestamp: int) -> datetime:
    datetime_obj = datetime.utcfromtimestamp(timestamp)
    return datetime(datetime_obj.year, datetime_obj.month, datetime_obj.day)


def _get_liwc_list_by_dt(subreddit: str, post_type: PostType, liwc_reddit_processor: LIWCRedditProcessor) \
        -> Dict[datetime, List[np.array]]:
    liwc_list_by_dt = defaultdict(list)
    assert post_type != PostType.ALL
    is_comment = post_type == PostType.COMMENT
    # ignore automoderator for comments (only)
    subr_data = get_subreddit_json(subreddit, post_type, ignore_automoderator=is_comment)
    for post in subr_data:
        post_text = get_text(post, include_title=True)
        features, word_count = liwc_reddit_processor.process_post(post["id"], is_comment, post_text)
        if word_count > 0:
            post_date = convert_timestamp(post["created_utc"])
            liwc_list_by_dt[post_date].append(features)
    return liwc_list_by_dt


def _compute_liwc_subreddit_csv_helper(subreddit: str, identifier: str,
                                       liwc_list_by_dt: Dict[datetime, List[np.array]],
                                       csv_columns: List[str]):
    file_path = f"{DATA_PATH}/{subreddit}_{identifier}.csv"

    subreddit_liwc_avg_df = pd.DataFrame(
        0, index=DatetimeIndex(pd.to_datetime(sorted(liwc_list_by_dt.keys()))),
        columns=csv_columns)
    for dt, liwc_feature_list in liwc_list_by_dt.items():
        n_datapoints = len(liwc_feature_list)
        document_liwc_avg = np.mean(liwc_feature_list, axis=0)
        data = np.concatenate((document_liwc_avg, [n_datapoints]))
        subreddit_liwc_avg_df.loc[dt] = data
    subreddit_liwc_avg_df.to_csv(file_path)


def compute_liwc_subreddit_csv(subreddit: str):
    # will create csv file containing avg LIWC features by day for posts, comments, posts + comments
    liwc_reddit_processor = LIWCRedditProcessor(include_title=True)

    # fetch post and comment dictionaries
    post_liwc_list_by_dt = _get_liwc_list_by_dt(subreddit, PostType.POST, liwc_reddit_processor)

    # create CSV file with averages by day
    csv_columns = liwc_reddit_processor.liwc_processor.liwc_categories + ["n_datapoints"]
    _compute_liwc_subreddit_csv_helper(subreddit, "post", post_liwc_list_by_dt, csv_columns)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddits",
                        nargs="+",
                        required=True,
                        help="Subreddit(s) to process")
    return parser.parse_args()


def main():
    args = _parse_args()
    for subreddit in tqdm(args.subreddits):
        compute_liwc_subreddit_csv(subreddit.lower())


if __name__ == "__main__":
    main()
