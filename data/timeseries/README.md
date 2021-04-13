# timeseries
This directory contains the timeseries used in our analysis. We have kept the timeseries that were used for final Prophet analysis and for graphing.

* The directory `journal/raw` contains the COVID lexicon timeseries, which were not normalized because we did not look at differences directly or perform prophet analysis.
* The directory `journal/normalized` contains the timeseries for our user interaction (graph), liwc, post count, and topic features. These are normalized. We include the individual series for each of our three mental health subreddits (r/Anxiety, r/depression, r/SuicideWatch), along with the average for the control subreddits. We also include the series representing the difference between each mental health subreddit and the controls.