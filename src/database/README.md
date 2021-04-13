# Database
This directory contains scripts related to the postgres database use for storing COVID lexicon/LIWC features.
* `create_tables.sql`: script to create tables to store features.  
  Notes: 
  * The table names may need dates appended to match the format in the `covid_postgres`/`liwc_postgres` scripts.
  * This code expects that you have a postgres database called `reddit_features` already created.
  * If you plan to do any other analysis, you may want to create additional indices, for instance on the subreddit/author columns.
* `postgres_connect.py`: script to connect to a postgres database which is used to store LIWC and COVID lexicon features