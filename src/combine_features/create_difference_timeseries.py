"""
Script to compute the difference between two existing timeseries
"""
import argparse
import os

from src.data_utils import read_subreddit_timeseries


def create_timeseries_difference(timeseries_path: str, series_name_1: str, series_name_2: str, start_date: str,
                                 end_date: str):
    """
    Combine together the timeseries for a given list of subreddits by averaging for each date
    """

    series_1 = read_subreddit_timeseries(timeseries_path, series_name_1)[start_date:end_date]
    series_2 = read_subreddit_timeseries(timeseries_path, series_name_2)[start_date:end_date]
    # make sure there are no missing dates in either series
    assert series_1.index.to_list() == series_2.index.to_list()
    diff_timeseries = series_1.sub(series_2)
    diff_timeseries.to_csv(
        os.path.join(timeseries_path, f"{series_name_1}_{series_name_2}_difference.csv"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series_name_1",
                        required=True,
                        help="Can be either the name of a subreddit or the name of a combined series.")
    parser.add_argument("--series_name_2",
                        required=True,
                        help="Can be either the name of a subreddit or the name of a combined series.")
    parser.add_argument("--timeseries_path",
                        required=True,
                        help="Path to existing timeseries/output path")
    parser.add_argument("--start_date",
                        default="2017",
                        help="The start date - can be passed in a string and will be parsed by pandas")
    parser.add_argument("--end_date",
                        default="2020-05",
                        help="The end date - can be passed in a string and will be parsed by pandas")
    return parser.parse_args()


def main():
    """
    Combine the two timeseries by computing the difference
    """
    args = _parse_args()
    create_timeseries_difference(args.timeseries_path, args.series_name_1, args.series_name_2,
                                 args.start_date, args.end_date)


if __name__ == "__main__":
    main()

