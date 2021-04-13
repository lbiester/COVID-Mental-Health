# post_count
This directory contains a single script to extract daily post counts as a time series. Usage:
```bash
PYTHONPATH=. python3 src/post_count/extract_post_count.py 
    --subreddits <list of subreddits, specify this or subreddit_file>
    --subreddit_file <path to subreddit file, specify this or subreddits>
    --output_path <path to output, we used data/timeseries/journal/raw/post_count>
```
The features can then be normalized and combined for Prophet modeling with the scripts in `src/combine_features`