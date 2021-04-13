# read some arguments from txt config file
import configparser

cfg = configparser.ConfigParser()
cfg.read("config.txt")
# read data locations
SUBREDDITS_DIR = cfg["DATA_LOCATIONS"]["SUBREDDITS_DIR"]
MONTHLY_DUMPS_DIR = cfg["DATA_LOCATIONS"]["MONTHLY_DUMPS_DIR"]
USERS_DIR = cfg["DATA_LOCATIONS"]["USERS_DIR"]
LIWC_2015_PATH = cfg["DATA_LOCATIONS"]["LIWC_2015_PATH"]

DUMP_AVAILABLE_POSTS = cfg["FILE_DUMPS"]["DUMP_AVAILABLE_POSTS"]
DUMP_AVAILABLE_COMMENTS = cfg["FILE_DUMPS"]["DUMP_AVAILABLE_COMMENTS"]
USE_DUMPS = cfg["FILE_DUMPS"]["USE_DUMPS"].lower() == "true"

MODERATORS_FILE_NAME = "MODERATORS"

LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
