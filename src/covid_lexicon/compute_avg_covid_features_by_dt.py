import argparse
import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from src.covid_lexicon.covid_lexicon_processor import CovidLexiconProcessor
from src.data_utils import get_subreddit_json, read_file_by_lines
from src.enums import PostType
from src.liwc.compute_avg_liwc_features_by_dt import convert_timestamp


def _process(subreddit: str, post_type: PostType, timeseries_path: str, by_post: bool):
    clp = CovidLexiconProcessor()
    all_features = []
    is_comment = post_type == PostType.COMMENT
    for post in get_subreddit_json(subreddit, post_type, properties=["id", "created_utc"],
                                   start_time=datetime(2019, 12, 25), end_time=datetime(2020, 6, 1),
                                   ignore_automoderator=is_comment):
        features, _ = clp.read_post_by_id(post["id"], is_comment)
        all_features.append((convert_timestamp(post["created_utc"]), features))

    post_times = [utc for utc, _ in all_features]
    post_features = [feat for _, feat in all_features]

    df_1 = pd.DataFrame(post_times, columns=["created_utc"])
    df_2 = pd.DataFrame(post_features, columns=clp.lexicon_words)
    df = pd.concat((df_1, df_2), axis=1)
    if by_post:
        daily_features = df.set_index("created_utc").sum(axis=1).apply(lambda x: int(x > 0)).resample("D").mean()
    else:
        # add to get counts for each word, then take the mean
        daily_features = df.set_index("created_utc").sum(axis=1).resample("D").mean()

    bp = "_bypost" if by_post else ""
    daily_features.to_csv(os.path.join(timeseries_path, f"{subreddit.lower()}_{post_type.name.lower()}{bp}.csv"))


def process(subreddit: str, timeseries_path: str, by_post: bool):
    _process(subreddit, PostType.POST, timeseries_path, by_post)
    _process(subreddit, PostType.COMMENT, timeseries_path, by_post)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddits",
                        nargs="+",
                        help="Subreddit(s) to process")
    parser.add_argument("--subreddit_file",
                        type=str,
                        help="Path to file with list of subreddits")
    parser.add_argument("--timeseries_path",
                        required=True,
                        help="Path to existing timeseries/output path")
    parser.add_argument("--by_post",
                        action="store_true",
                        help="Compute % of posts")
    args = parser.parse_args()
    if ((args.subreddits is not None and args.subreddit_file is not None)
            or (args.subreddits is None and args.subreddit_file is None)):
        parser.error("Must specify subreddits OR subreddit_file")
    return args


def main():
    # process arguments
    args = _parse_args()
    if args.subreddit_file:
        subreddits = read_file_by_lines(args.subreddit_file)
    else:
        subreddits = args.subreddits

    for subreddit in tqdm(subreddits):
        process(subreddit, args.timeseries_path, args.by_post)


if __name__ == "__main__":
    main()
