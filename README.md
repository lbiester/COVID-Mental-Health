# Quantifying the Effects of COVID-19 on Mental Health Support Forums
#### Laura Biester*, Katie Matton*, Janarthanan Rajendran, Emily Mower Provost, and Rada Mihalcea
This repository contains our code for the paper [_Quantifying the Effects of COVID-19 on Mental Health Support Forums_](https://www.aclweb.org/anthology/2020.nlpcovid19-2.8.pdf), 
which was presented at the NLP COVID-19 Workshop @EMNLP2020.

\* Laura Biester and Katie Matton contributed equally to this project.

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
2. Extract features using the code in `src/covid_lexicon`, `src/user_count`, `src/liwc`, or `src/topic`.
3. (If applicable) run prophet timeseries analysis using `src/time_series_model_prophet.py`
4. Generate results tables/graphs using the code in `src/results`
