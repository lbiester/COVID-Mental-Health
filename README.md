# Understanding the Impact of COVID-19 on Online Mental Health Forums
#### Laura Biester, Katie Matton, Janarthanan Rajendran, Emily Mower Provost, and Rada Mihalcea
This repository contains our code for the paper Understanding the Impact of COVID-19 on Online Mental Health Forums, 
which will soon appear in _ACM Transactions on Management Information Systems_.

An earlier version of this work was presented at the NLP COVID-19 Workshop @EMNLP2020. The code for this version of the 
paper is available under the [workshop release](https://github.com/lbiester/COVID-Mental-Health/releases/tag/workshop).

---

To run this code, you must first write a config.txt file with the following format:
```
[DATA_LOCATIONS]
SUBREDDITS_DIR = ?
USERS_DIR = ?
LIWC_2015_PATH = ?
```
The directories reference where subreddit and user JSON data is stored. This data is expected
to be stored in .jsonl files for each month for each subreddit, i.e. "202001.jsonl". 
It would be fairly simple to switch out the `src/data_utils.py` script to read using some other format if necessary.
We do not end up using user directory in this project.

The `LIWC_2015_PATH` references where the LIWC 2015 dictionary is on disk.

The LIWC and COVID lexicon code stores features in a postgres database. For this, you need two additional entries in 
config.txt:
```
[POSTGRES]
USER = ?
PASSWORD = ?
```
Reflecting your postgres username and password.

This is the general workflow to follow to run our pipeline
1. Make sure that you have your data ready. Data from the mental health subreddits is not provided in this repo.
2. Extract features using the code in `src/covid_lexicon`, `src/post_count`, `src/graph`, `src/liwc`, or `src/topic`.
3. Normalize and combine time series using the code in `src/combine_features`
4. (If applicable) run prophet timeseries analysis using `src/time_series_model_prophet.py`
5. Generate results tables/graphs using the code in `src/results`
