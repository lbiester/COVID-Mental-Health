import argparse
import os
from typing import List, Optional

import pandas as pd

from src.data_utils import read_subreddit_timeseries, read_file_by_lines


class SubredditTimeseriesCombiner:
    """
    Class to combine timeseries for a number of subreddits to a single average timeseries
    Average with subreddits equally weighted to reduce the impact of large subreddits
    """
    def __init__(self, timeseries_path):
        self.timeseries_path = timeseries_path

    def _read_subreddit_list_timeseries(self, subreddits: List[str], suffix: Optional[str]) -> List[pd.DataFrame]:
        """
        Create a list of subreddit timeseries dataframes
        """
        return [read_subreddit_timeseries(self.timeseries_path, f"{subreddit_name}{suffix}")
                for subreddit_name in subreddits]

    @staticmethod
    def _create_mean_dataframe(df_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Create a dataframe with average values for each day
        """
        concat_df = pd.concat(df_list)
        return concat_df.groupby(concat_df.index).mean()

    def combine_subreddit_timeseries(self, subreddits: List[str], subreddit_list_name: str, suffix: Optional[str]):
        """
        Combine together the timeseries for a given list of subreddits by averaging for each date
        """
        suffix = f"_{suffix}" if suffix is not None else ""
        # format output directory
        output_file = os.path.join(self.timeseries_path, f"{subreddit_list_name}{suffix}.csv")

        # read post, comment, and all dataframes
        df_list = self._read_subreddit_list_timeseries(subreddits, suffix)
        mean_df = self._create_mean_dataframe(df_list)
        mean_df.to_csv(output_file)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddits",
                        nargs="+",
                        help="Subreddit(s) to process")
    parser.add_argument("--subreddit_file",
                        type=str,
                        help="Path to file with list of subreddits")
    parser.add_argument("--subreddit_list_name",
                        required=True,
                        help="Name of the subreddit list")
    parser.add_argument("--suffix",
                        help="The suffix in the timeseries filenames")
    parser.add_argument("--timeseries_path",
                        required=True,
                        help="Path to existing timeseries/output path")
    args = parser.parse_args()
    if ((args.subreddits is not None and args.subreddit_file is not None)
            or (args.subreddits is None and args.subreddit_file is None)):
        parser.error("Must specify subreddits OR subreddit_file")
    return args


def main():
    """
    Create a SubredditTimeseriesCombiner and call it with the provided list of subreddits
    """
    # process arguments
    args = _parse_args()
    if args.subreddit_file:
        subreddits = read_file_by_lines(args.subreddit_file)
    else:
        subreddits = args.subreddits

    stc = SubredditTimeseriesCombiner(args.timeseries_path)
    stc.combine_subreddit_timeseries(subreddits, args.subreddit_list_name, args.suffix)


if __name__ == "__main__":
    main()
