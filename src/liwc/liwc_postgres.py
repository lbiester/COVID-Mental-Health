from typing import Tuple, Set, Optional

import base36
import numpy as np
from psycopg2.extensions import cursor

from src import config
from src.database.postgres_connect import connect_psql

# determine table name based on availability and use of post/comment files
dap = config.DUMP_AVAILABLE_POSTS
dac = config.DUMP_AVAILABLE_COMMENTS

# format of table name: base name + _ + last post dump date + _ + last comment dump date
TABLE_NAME = "liwc_features_covid_include_title_custom_tokenize" if not config.USE_DUMPS \
    else f"liwc_features_covid_include_title_custom_tokenize_{dap}_{dac}"


READ_IDS_STMT = f"SELECT reddit_id, comment from {TABLE_NAME }"
READ_BY_ID_STMT = f"SELECT * FROM {TABLE_NAME} " \
                  f"WHERE reddit_id = {{}} and comment = {{}}"
INSERT_COMMAND_FMT = f"INSERT INTO {TABLE_NAME} VALUES ({{}}) ON CONFLICT DO NOTHING"
EXISTS_STMT = f"SELECT 1 FROM {TABLE_NAME} WHERE reddit_id = {{}} AND comment = {{}}"


def read_all_ids() -> Set[Tuple[str, bool]]:
    id_set = set()
    connection = connect_psql()
    csr = connection.cursor()
    csr.execute(READ_IDS_STMT)
    while True:
        # do something with row
        row = csr.fetchone()
        if row is None:
            break
        id_set.add(row)
    return {(base36.dumps(base10_id), is_comment) for base10_id, is_comment in id_set}


def read_by_id(base36_id: str, is_comment: bool, csr: cursor) -> Optional[np.array]:
    base10_id = base36.loads(base36_id)
    read_stmt = READ_BY_ID_STMT.format(base10_id, is_comment)
    csr.execute(read_stmt)
    row = csr.fetchone()
    if row is not None:
        return np.array(row[2:-3], dtype=np.uint16)


def exists_in_db(base36_id: str, is_comment: bool, csr: cursor) -> bool:
    base10_id = base36.loads(base36_id)
    exists_stmt = EXISTS_STMT.format(base10_id, is_comment)
    csr.execute(exists_stmt)
    row = csr.fetchone()
    return row is not None


def save_to_database(base36_id: str, is_comment: bool, created_utc: int, subreddit: str, author: str,
                     liwc_features: np.array, csr: cursor):
    base10_id = base36.loads(base36_id)
    insert_cmd = INSERT_COMMAND_FMT.format(f"{base10_id}, {is_comment}, {','.join(str(x) for x in liwc_features)}, "
                                           f"to_timestamp({created_utc}), '{subreddit.lower()}', '{author.lower()}'")
    csr.execute(insert_cmd)
