# src
This directory contains all of our project code. Each sub-directory has a README.
The files here are as follows

* `config.py`: project configuration, such as where data is stored and logging format
* `data_utils.py`: functions for reading reddit data from disk
* `enums.py`: some enums, i.e. PostType (which can be POST or COMMENT). In the workshop paper, we focus exclusively on posts.
* `stat_utils.py`: function for performing our significance test
* `text_utils.py`: functions for text pre-processing
* `time_series_model_prophet.py`: script for prophet timeseries modeling