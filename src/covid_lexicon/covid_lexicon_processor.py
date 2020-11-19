import argparse
import logging
from itertools import islice
from typing import List, Iterable, Tuple

import numpy as np
from argparse_utils import enum_action
from tqdm import tqdm

from src import config
from src.database.postgres_connect import connect_psql
from src.enums import PostType, EntityType
from src.covid_lexicon.covid_postgres import read_by_id, save_to_database, exists_in_db, get_feature_names, \
    should_process
from src.data_utils import get_text, get_all_scraped_subreddits, get_all_scraped_users, get_entity_json, \
    read_file_by_lines

# logging
from src.liwc.liwc_processor import liwc_tokenize

logging.basicConfig(format=config.LOGGING_FORMAT)
logger = logging.getLogger(__file__)


class CovidLexiconProcessor:
    # ignores case
    SAVE_DATA_FREQUENCY = 10000
    REDDIT_ID_COL = "reddit_id"
    IS_COMMENT_COL = "comment"
    MAX_ENTITIES_PER_CALL = 5000

    def __init__(self, include_title=True):
        self.to_save = {}
        self.post_count = 0
        self.include_title = include_title
        self.cursor = connect_psql().cursor()
        self.lexicon_words = read_file_by_lines("data/covid_lexicon/covid19-unigrams.txt")
        assert get_feature_names(self.cursor) == self.lexicon_words
        self.lexicon_word_to_idx = {word: i for i, word in enumerate(sorted(self.lexicon_words))}
        self.n_features = len(self.lexicon_words)
        self.processed_count = 0

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
        while True:
            next_objs = list(islice(entity_json_iterable, self.MAX_ENTITIES_PER_CALL))
            if len(next_objs) == 0:
                break
            post_queue = should_process(next_objs, post_type == PostType.COMMENT, self.cursor)
            self.process_posts(post_queue, post_type == PostType.COMMENT)

    def process_posts(self, post_list: List[dict], is_comment: bool):
        for post in post_list:
            # get counts
            lexicon_counts = self.count_lexicon_words(get_text(post, include_title=self.include_title))
            # save_to_db
            save_to_database(post["id"], is_comment, post["subreddit"], post["created_utc"], post["author"],
                             lexicon_counts, self.cursor)
        self.save_data(items_processed=len(post_list))
        logger.debug("Processed %d posts", len(post_list))

    def read_post_by_id(self, post_id: str, is_comment: bool, length_normalize: bool = True) \
            -> Tuple[np.array, int]:
        raw_features = read_by_id(post_id, is_comment, self.cursor)
        if raw_features is None:
            raise Exception("Must process post before calling read_post")
        liwc_features = raw_features[:self.n_features]
        word_count = raw_features[self.n_features]
        if length_normalize and word_count != 0:
            # normalize so that counts divided by total word count
            liwc_features = liwc_features / word_count

        return liwc_features, word_count

    def save_data(self, items_processed: int = 1, force: bool = False) -> None:
        self.processed_count += items_processed
        if force or self.processed_count >= self.SAVE_DATA_FREQUENCY:
            logger.info("Saving COVID data to database")
            self.cursor.connection.commit()
            logger.info("Done saving COVID database")
            self.processed_count = 0

    def count_lexicon_words(self, post_text: str) -> np.array:
        result_array = np.zeros(len(self.lexicon_words) + 1, dtype=np.uint16)
        tokenized_post = liwc_tokenize(post_text)
        for word in tokenized_post:
            if word in self.lexicon_words:
                result_array[self.lexicon_word_to_idx[word]] += 1
        result_array[-1] = len(tokenized_post)
        return result_array

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
    covid_lexicon_parser = CovidLexiconProcessor(include_title=True)
    if args.subreddit or args.subreddit_all:
        subreddits = args.subreddit or sorted(get_all_scraped_subreddits())
        covid_lexicon_parser.process_subreddits(subreddits, args.post_type)
    if args.user or args.user_all or args.user_list:
        if args.user_all:
            users = sorted(get_all_scraped_users())
        elif args.user_list:
            users = sorted(read_file_by_lines(args.user_list))
        else:
            users = args.user
        covid_lexicon_parser.process_users(users, args.post_type)
    covid_lexicon_parser.save_data(force=True)


if __name__ == "__main__":
    main()

