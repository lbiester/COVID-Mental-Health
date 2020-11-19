# LIWC
These scripts are used for extracting LIWC features
* `compute_avg_liwc_features_by_dt.py`: retrieves pre-computed LIWC values from postgres and outputs averages for each
  date in a CSV file
* `liwc_postgres.py`: stores LIWC counts in a postgres database
* `liwc_processor.py`: computes LIWC values from dictionary
* `liwc_reddit_processor.py`: goes through Reddit posts and computes LIWC values

To get the LIWC values used in this project, start by running `liwc_reddit_processor.py`:
```bash

PYTHONPATH=. python3 src/liwc/liwc_reddit_processor.py --subreddit <subreddit you are computing values for>
                                                       --post_type POST
```

Then, get the averages for each subreddit using `compute_avg_liwc_features_by_dt.py`:
```bash
PYTHONPATH=. python3 src/liwc/compute_avg_liwc_features_by_dt.py --subreddits anxiety depression suicidewatch
```