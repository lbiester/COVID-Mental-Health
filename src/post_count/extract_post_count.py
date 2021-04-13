"""
Script to get post counts
Ignore the automoderator for comments, based on activity in the anxiety subreddit around covid
Keep removed data for counts
"""
import argparse
import os
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from src.data_utils import read_file_by_lines, get_subreddit_json
from src.enums import PostType


def extract_post_count(subreddit: str, post_type: PostType):
    daily_post_counts = Counter()
    posts = get_subreddit_json(subreddit, post_type, start_time=datetime(2017, 1, 1), end_time=datetime(2020, 6, 1),
                               properties=["created_utc"], ignore_automoderator=(post_type == PostType.COMMENT),
                               filter_removed=False)
    for post in posts:
        dt = datetime.fromtimestamp(post["created_utc"], timezone.utc).date()
        daily_post_counts[dt] += 1
    ser = pd.Series(daily_post_counts, name=f"{post_type.name.lower()}_count")
    return ser


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddits",
                        nargs="+",
                        help="Subreddit(s) to process")
    parser.add_argument("--subreddit_file",
                        type=str,
                        help="Path to file with list of subreddits")
    parser.add_argument("--output_path",
                        required=True,
                        help="Path to timeseries output path")
    args = parser.parse_args()
    if ((args.subreddits is not None and args.subreddit_file is not None)
            or (args.subreddits is None and args.subreddit_file is None)):
        parser.error("Must specify subreddits OR subreddit_file")
    return args


def main():
    args = _parse_args()
    if args.subreddit_file:
        subreddits = read_file_by_lines(args.subreddit_file)
    else:
        subreddits = args.subreddits

    for subreddit in subreddits:
        print("processing posts")
        post_series = extract_post_count(subreddit, PostType.POST)
        print("processing comments")
        comment_series = extract_post_count(subreddit, PostType.COMMENT)
        df = pd.concat((post_series, comment_series), axis=1)
        df.to_csv(os.path.join(args.output_path, f"{subreddit.lower()}.csv"))


if __name__ == "__main__":
    main()
