import re
from typing import Optional, List

import base36
import numpy as np
from psycopg2.extensions import cursor

READ_BY_ID_STMT = "SELECT * FROM covid_term_count WHERE reddit_id = {} and comment = {}"
INSERT_COMMAND_FMT = "INSERT INTO covid_term_count VALUES ({}) ON CONFLICT DO NOTHING"
EXISTS_STMT = "select 1 from covid_term_count WHERE reddit_id = {} and comment = {}"
GET_COLUMNS_STMT = "SELECT column_name FROM information_schema.columns WHERE table_catalog = 'reddit_features' " \
                   "and table_name = 'covid_term_count' ORDER BY ordinal_position"
EXISTS_MULTIPLE_STMT = "SELECT reddit_id FROM covid_term_count WHERE reddit_id IN ({}) and comment = {}"

# columns used for searching (as opposed to storing count data)
SEARCH_COLUMNS = ["reddit_id", "comment", "subreddit", "date", "author"]
WORD_COUNT_COL = "word_count"



def get_feature_names(csr: cursor) -> List[str]:
    # used to confirm that columns match our lexicon
    csr.execute(GET_COLUMNS_STMT)
    column_names = [x[0] for x in csr.fetchall()]
    assert set(SEARCH_COLUMNS).issubset(set(column_names))
    assert WORD_COUNT_COL in column_names
    feature_columns = [col for col in column_names if col != WORD_COUNT_COL and col not in SEARCH_COLUMNS]
    feature_names = [re.sub("^count_", "", col) for col in feature_columns]
    feature_names_clean = [re.sub("_", "-", col) for col in feature_names]
    return feature_names_clean


def read_by_id(base36_id: str, is_comment: bool, csr: cursor) -> Optional[np.array]:
    base10_id = base36.loads(base36_id)
    read_stmt = READ_BY_ID_STMT.format(base10_id, is_comment)
    csr.execute(read_stmt)
    row = csr.fetchone()
    if row is not None:
        return np.array(row[len(SEARCH_COLUMNS):], dtype=np.uint16)


def exists_in_db(base36_id: str, is_comment: bool, csr: cursor) -> bool:
    base10_id = base36.loads(base36_id)
    exists_stmt = EXISTS_STMT.format(base10_id, is_comment)
    csr.execute(exists_stmt)
    row = csr.fetchone()
    return row is not None


def save_to_database(base36_id: str, is_comment: bool, subreddit: str, created_utc: int, author: str,
                     covid_features: np.array, csr: cursor):
    base10_id = base36.loads(base36_id)
    insert_cmd = INSERT_COMMAND_FMT.format(f"{base10_id}, {is_comment}, '{subreddit.lower()}', '{author.lower()}',"
                                           f"to_timestamp({created_utc}), {','.join(str(x) for x in covid_features)}")
    csr.execute(insert_cmd)


def should_process(post_list: List[dict], is_comment: bool, csr: cursor) -> List[dict]:
    # returns a list of posts that should actually be processed out of the list (because they have not been yet)
    base10_ids = [base36.loads(post["id"]) for post in post_list]
    exists_multiple_stmt = EXISTS_MULTIPLE_STMT.format(",".join(str(x) for x in base10_ids), is_comment)
    csr.execute(exists_multiple_stmt)
    exclude_ids = set(base36.dumps(x[0]) for x in csr.fetchall())
    should_process_list = [post for post in post_list if post["id"] not in exclude_ids]
    return should_process_list