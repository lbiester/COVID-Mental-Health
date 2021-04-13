import argparse
import logging
from typing import List, Iterable, Tuple, Dict

import numpy as np
from argparse_utils import enum_action
from tqdm import tqdm

from src import config
from src.data_utils import get_text, get_all_scraped_subreddits, get_all_scraped_users, get_entity_json, \
    read_file_by_lines
from src.database.postgres_connect import connect_psql
from src.enums import PostType, EntityType
from src.liwc.liwc_postgres import read_by_id, save_to_database, exists_in_db
from src.liwc.liwc_processor import LIWCProcessor

# logging
logging.basicConfig(format=config.LOGGING_FORMAT)
logger = logging.getLogger(__file__)


class LIWCRedditProcessor:
    SAVE_DATA_FREQUENCY = 10000
    REDDIT_ID_COL = "reddit_id"
    IS_COMMENT_COL = "comment"
    MAX_ENTITIES_PER_CALL = 100

    def __init__(self, include_title=False):
        self.liwc_processor = LIWCProcessor(custom_tokenize=True, ignore_case=True)
        self.to_save = {}
        self.post_metadata = {}
        self.post_count = 0
        self.include_title = include_title
        self.cursor = connect_psql().cursor()

    def process_users(self, users: List[str], post_type: PostType) -> None:
        if post_type == PostType.ALL or post_type == PostType.POST:
            self.process_entities(EntityType.USER, users, PostType.POST)
        if post_type == PostType.ALL or post_type == PostType.COMMENT:
            self.process_entities(EntityType.USER, users, PostType.COMMENT)

    def process_subreddits(self, subreddits: List[str], post_type: PostType):
        if post_type == PostType.ALL or post_type == PostType.POST:
            self.process_entities(EntityType.SUBREDDIT, subreddits, PostType.POST)
        if post_type == PostType.ALL or post_type == PostType.COMMENT:
            self.process_entities(EntityType.SUBREDDIT, subreddits, PostType.COMMENT)

    def process_entities(self, entity_type: EntityType, entity_names: List[str], post_type: PostType) -> None:
        for entity_name in tqdm(entity_names, desc="processed entites"):
            logger.info("Processing entity %s, %s", entity_name, post_type.name.lower())
            entity_json_iterable = get_entity_json(
                    entity_type, entity_name, post_type)
            self._process_entities_helper(entity_json_iterable, post_type)

    def _process_entities_helper(self, entity_json_iterable: Iterable[dict], post_type: PostType) -> None:
        more_data = True
        while more_data:
            post_queue = []
            while more_data and len(post_queue) < self.MAX_ENTITIES_PER_CALL:
                next_obj = next(entity_json_iterable, None)
                if next_obj is None:
                    more_data = False
                    break
                if not self._is_empty(next_obj) and not self._in_db(next_obj, post_type):
                    post_queue.append(next_obj)
            self.process_posts(post_queue, post_type == PostType.COMMENT)

    def process_posts(self, post_list: List[dict], is_comment: bool) -> Dict[Tuple[str, bool], np.array]:
        post_text_list = [get_text(post, include_title=self.include_title) for post in post_list]
        result_list = []
        for post in post_text_list:
            result_list.append(self.liwc_processor.vectorize(post))

        # set values
        assert len(result_list) == len(post_list)
        return_dict = {}
        for post, result in zip(post_list, result_list):
            self.to_save[(post["id"], is_comment)] = result
            self.post_metadata[(post["id"], is_comment)] = {k: v for k, v in post.items()
                                                            if k in {"author", "subreddit", "created_utc"}}
            return_dict[(post["id"], is_comment)] = result

        # save results
        self.save_data()
        logger.debug("Processed %d posts", len(post_list))

        # return dictionary with just this set of processed posts
        return return_dict

    def process_post(self, post_id: str, is_comment: bool, text: str, author: str, subreddit: str, created_utc: int,
                     length_normalize: bool = True) -> Tuple[np.array, int]:
        if (post_id, is_comment) in self.to_save:
            raw_features = self.to_save[(post_id, is_comment)]
        else:
            raw_features = read_by_id(post_id, is_comment, self.cursor)
            if raw_features is None:
                raw_features = self.liwc_processor.vectorize(text)
                self.to_save[(post_id, is_comment)] = raw_features
                self.post_metadata[(post_id, is_comment)] = {
                    "author": author, "subreddit": subreddit, "created_utc": created_utc}
                self.save_data()
        n_features = len(self.liwc_processor.liwc_categories)
        liwc_features = raw_features[:n_features]
        word_count = raw_features[n_features]
        if length_normalize and word_count != 0:
            # normalize so that counts divided by total word count
            liwc_features = liwc_features / word_count

        return liwc_features, word_count

    def save_data(self, force: bool = False) -> None:
        if force or len(self.to_save) >= self.SAVE_DATA_FREQUENCY:
            # convert to dataframe
            logger.info("Saving LIWC data to database")
            # save to post to features to database
            for key, value in self.to_save.items():
                save_to_database(key[0], key[1], self.post_metadata[key]["created_utc"],
                                 self.post_metadata[key]["subreddit"], self.post_metadata[key]["author"], value,
                                 self.cursor)
            self.to_save = {}
            self.post_metadata = {}
            self.cursor.connection.commit()
            logger.info("Done saving LIWC database")

    @staticmethod
    def _convert_to_csv_line(key, value):
        return ",".join([str(k) for k in key]) + "," + ",".join([str(v) for v in value]) + "\n"

    @staticmethod
    def _infer_post_type(post: dict) -> PostType:
        assert not (("title" in post or "body" in post) and "selftext" in post)
        if "title" in post or "body" in post:
            return PostType.POST
        else:
            return PostType.COMMENT

    def _is_empty(self, post: dict) -> bool:
        return len(get_text(post, include_title=self.include_title)) == 0

    def _in_db(self, post: dict, post_type: PostType) -> bool:
        return exists_in_db(post["id"], post_type == PostType.COMMENT, self.cursor)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subreddit", dest="subreddit", nargs="+", help="Subreddits to process")
    parser.add_argument("-sa", "--subreddit_all", dest="subreddit_all", action="store_true",
                        help="Process all subreddits")
    parser.add_argument("-u", "--user", dest="user", nargs="+", help="Users to process")
    parser.add_argument("-ua", "--user_all", dest="user_all", action="store_true", help="Process all users")
    parser.add_argument("-ul", "--user_list", dest="user_list", type=str, help="List of users to process")
    parser.add_argument("-pt", "--post_type", dest="post_type", action=enum_action(PostType), default=PostType.ALL,
                        help="Post type to process")
    parser.add_argument("--log", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO",
                        help="Set the logging level")

    args = parser.parse_args()

    if sum(map(bool, [args.subreddit, args.user, args.subreddit_all, args.user_all, args.user_list])) != 1:
        # this is basically xoring the 4
        parser.error("Must provide one argument to select user(s) or subreddit(s)")
    # initialize logging at correct level
    if args.log:
        logging.getLogger().setLevel(getattr(logging, args.log))

    return args


def main():
    args = _parse_args()
    liwc_reddit_processor = LIWCRedditProcessor(include_title=True)
    if args.subreddit or args.subreddit_all:
        subreddits = args.subreddit or sorted(get_all_scraped_subreddits())
        liwc_reddit_processor.process_subreddits(subreddits, args.post_type)
    if args.user or args.user_all or args.user_list:
        if args.user_all:
            users = sorted(get_all_scraped_users())
        elif args.user_list:
            users = sorted(read_file_by_lines(args.user_list))
        else:
            users = args.user
        liwc_reddit_processor.process_users(users, args.post_type)
    liwc_reddit_processor.save_data(force=True)


if __name__ == "__main__":
    main()

