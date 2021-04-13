"""
Script to normalize a timeseries
"""
import argparse
import os
from datetime import date, datetime

from argparse_utils import date_action
from dateutil.relativedelta import relativedelta

from src.data_utils import read_subreddit_timeseries


def normalize_timeseries(timeseries_path: str, series_name: str, start_date: str,
                         prediction_start: datetime, end_date: str, output_path: str):
    """
    Combine together the timeseries for a given list of subreddits by averaging for each date
    """
    # format output directory
    orig_series = read_subreddit_timeseries(timeseries_path, series_name)[start_date:end_date]

    # mean and standard deviation are computed with pre-covid data
    mean_series = orig_series[:prediction_start - relativedelta(days=1)].mean()
    std_series = orig_series[:prediction_start - relativedelta(days=1)].std()

    normalized_series = (orig_series - mean_series) / std_series

    # replace NA with 0. This notably happens is necessary if all of the values are zero.
    normalized_series.fillna(0, inplace=True)

    normalized_series.to_csv(os.path.join(output_path, f"{series_name.lower()}.csv"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series_name",
                        required=True,
                        help="Can be either the name of a subreddit or the name of a combined series. "
                             "Should include a suffix.")
    parser.add_argument("--timeseries_path",
                        required=True,
                        help="Path to existing timeseries/output path")
    parser.add_argument("--output_path",
                        required=True,
                        help="Path to save resulting timeseries")
    parser.add_argument("--start_date",
                        default="2017",
                        help="The start date - can be passed in a string and will be parsed by pandas")
    parser.add_argument("--prediction_start",
                        action=date_action(fmt="%Y-%m-%d"),
                        default=date(2020, 3, 1),
                        help="Date of intervention "
                             "(default = beginning of when we see an effect of COVID on MH subreddits)")
    parser.add_argument("--end_date",
                        default="2020-05",
                        help="The end date - can be passed in a string and will be parsed by pandas")
    return parser.parse_args()


def main():
    """
    Combine the two timeseries by creating a ratio
    """
    args = _parse_args()
    normalize_timeseries(args.timeseries_path, args.series_name, args.start_date, args.prediction_start,
                         args.end_date, args.output_path)


if __name__ == "__main__":
    main()
