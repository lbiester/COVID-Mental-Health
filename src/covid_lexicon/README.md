# covid_lexicon
These scripts are used for extracting COVID lexicon features
* `covid_postgres.py`: stores COVID lexicon counts in a postgres database
* `covid_lexicon_processor.py`: goes through Reddit posts and computes COVID lexicon values
* `compute_avg_covid_features_by_dt.py`: computes the average daily COVID feature values

To get the COVID lexicon values used in this project, run `covid_lexicon_processor.py`:
```bash
PYTHONPATH=. python3 src/covid_lexicon/covid_lexicon_processor.py --subreddit <subreddit you are computing values for>
                                                                  --post_type POST
```

Then, the final graphs we produce can be created using `src/results/generate_covid_discussion_graph.py`