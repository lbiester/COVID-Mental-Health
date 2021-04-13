# combine_features
These scripts are used to combine features from subreddits for timeseries analysis.
* `normalize_timeseries.py`: normalize time series values
* `combine_subreddit_timeseries.py`: average values across subreddits
* `create_difference_timeseries.py`: compute the difference between target and control subreddits

Start by normalizing values as follows for each subreddit
```bash
PYTHONPATH=. python3 src/combine_features/normalize_timeseries.py 
    --series_name <name of the series you are using, for instance anxiety_post>
    --timeseries_path <input time series path, in our case data/timeseries/raw/{feature_type}>
    --output_path <output time series path, in our case data/timeseries/normalized/{feature_type}>
```

Next, combine the control subreddits
```bash
PYTHONPATH=. python3 src/combine_features/combine_subreddit_timeseries.py 
    --subreddit_file data/subreddit_lists/baseline_subreddits/baseline_subreddits.txt
    --subreddit_list_name <name to use for subreddit list in output, we use "baseline_avg">
    --suffix <optional suffix, used in topics/graph naming convention>
    --timeseries_path <time series  path, in our case data/timeseries/normalized/{feature_type}>
```

Finally, compute the difference between the target and control series
```bash
PYTHONPATH=. python3 src/combine_features/create_difference_timeseries.py 
    --series_name_1 <name of target series>
    --series_name_2 <name of control series>
    --timeseries_path <time series  path, in our case data/timeseries/normalized/{feature_type}>
```