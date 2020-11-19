"""
Script to generate Table 1
"""
from datetime import datetime
from typing import Dict

import pandas as pd

from src.data_utils import get_subreddit_json
from src.enums import PostType


def get_counts(subreddit: str) -> Dict[str, float]:
    """
    Function that gets post counts for 2017 - 2020 (up through May) for each subreddit
    :param subreddit: The subreddit to get counts from
    :return: A dictionary of year --> avg count in that year
    """
    subreddit_counts = []
    for post in get_subreddit_json(subreddit, PostType.POST, start_time=datetime(2017, 1, 1),
                                   end_time=datetime(2020, 6, 1), ignore_automoderator=False):
        subreddit_counts.append((post["created_utc"], post["author"]))

    df = pd.DataFrame(subreddit_counts, columns=["date", "author"])
    df["date"] = pd.to_datetime(df["date"], unit="s")
    by_dt = df.set_index("date")

    post_counts = {
        "2017": by_dt.loc["2017"].resample("D").count().mean().item(),
        "2018": by_dt.loc["2018"].resample("D").count().mean().item(),
        "2019": by_dt.loc["2019"].resample("D").count().mean().item(),
        "2020": by_dt.loc["2020"].resample("D").count().mean().item()
    }

    return post_counts


def main():
    """
    Get all counts, create a dataframe, and display them
    """
    all_results = {
        "r/Anxiety": get_counts("Anxiety"),
        "r/depression": get_counts("depression"),
        "r/SuicideWatch": get_counts("SuicideWatch")
    }
    print(pd.DataFrame.from_dict(all_results))


if __name__ == "__main__":
    main()
