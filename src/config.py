# read some arguments from txt config file
import configparser

cfg = configparser.ConfigParser()
cfg.read("config.txt")
# read data locations
SUBREDDITS_DIR = cfg["DATA_LOCATIONS"]["SUBREDDITS_DIR"]
USERS_DIR = cfg["DATA_LOCATIONS"]["USERS_DIR"]
LIWC_2015_PATH = cfg["DATA_LOCATIONS"]["LIWC_2015_PATH"]

MODERATORS_FILE_NAME = "MODERATORS"

LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"