import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil import rrule

from src import data_utils
from src.enums import PostType

JAN_1_2017 = data_utils.get_utc_timestamp(datetime(2017, 1, 1))
JUN_1_2020 = data_utils.get_utc_timestamp(datetime(2020, 6, 1))
OUTPUT_DIR = "data/timeseries/user_count/"


def compute_all_posts_by_user_subreddit(posts_by_subreddit):
    all_posts_by_user_subreddit = defaultdict(dict)
    for subreddit, posts in posts_by_subreddit.items():
        for post in posts:
            created_utc = post["created_utc"]
            user = post["author"]
            if user not in all_posts_by_user_subreddit[subreddit]:
                all_posts_by_user_subreddit[subreddit][user] = []
            all_posts_by_user_subreddit[subreddit][user].append(created_utc)

    return all_posts_by_user_subreddit


def n_user_series(granularity, all_posts_by_user_subreddit, min_timestamp, max_timestamp):
    """Number of users who have posted between the given duration"""
    bins = [data_utils.get_utc_timestamp(dt)
            for dt in rrule.rrule(granularity, dtstart=datetime.utcfromtimestamp(min_timestamp),
                                  until=datetime.utcfromtimestamp(max_timestamp))]
    n_users = {}
    for subreddit in all_posts_by_user_subreddit:
        n_users[subreddit] = [0] * (len(bins) - 1)
        for user in all_posts_by_user_subreddit[subreddit]:
            posts_hist, _ = np.histogram(all_posts_by_user_subreddit[subreddit][user], bins)
            n_users[subreddit] += np.where(posts_hist > 0, 1, 0)

    return bins, n_users


def main():
    posts_by_subreddit = {}
    for subreddit in data_utils.get_mh_subreddits():
        posts_by_subreddit[subreddit] = list(data_utils.get_subreddit_json(subreddit, PostType.POST,
                                                                           properties=["author", "created_utc"]))

    all_posts_by_user_subreddit = compute_all_posts_by_user_subreddit(posts_by_subreddit)
    bins, n_users = n_user_series(rrule.DAILY, all_posts_by_user_subreddit, JAN_1_2017, JUN_1_2020)

    # write data to disk
    for subreddit, user_count_series in n_users.items():
        index = pd.date_range(datetime(2017, 1, 1), datetime(2020, 5, 31))
        ser = pd.Series(index=index, data=user_count_series, name="user_count")
        ser.to_csv(os.path.join(OUTPUT_DIR, f"{subreddit.lower()}_post.csv"))


if __name__ == "__main__":
    main()
