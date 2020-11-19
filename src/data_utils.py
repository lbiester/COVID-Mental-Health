import datetime
import itertools
import json
import os
import re
from typing import Any, Set, List, Dict, Iterable, Iterator, Tuple, Union, Optional

import pandas as pd
from dateutil.relativedelta import relativedelta

from src import config
from src.enums import EntityType, PostType


def read_file_by_lines(filename: str) -> List[str]:
    """
    Read a file into a list of lines
    """
    with open(filename, "r") as f:
        return f.read().splitlines()


def save_iterable_to_file_by_lines(iterable: Iterable[Any], filename: str):
    """
    Write data from an iterable to a file, line by line
    Converts to string type
    """
    with open(filename, "w") as f:
        for line in iterable:
            f.write(f"{line}\n")


def post_type_str_to_enum(post_type_str: str) -> PostType:
    """
    Convert string specifying post type to enum.
    """
    post_type = PostType.ALL
    if post_type_str == "post":
        post_type = PostType.POST
    elif post_type_str == "comment":
        post_type = PostType.COMMENT
    return post_type


def get_mh_subreddits() -> List[str]:
    """
    Get a list of mental health subreddits from a file
    """
    return read_file_by_lines("data/mh_subreddits.txt")


def get_coronavirus_subreddits() -> List[str]:
    """
    Get a list of mental health subreddits from a file
    """
    return read_file_by_lines("data/coronavirus_subreddits.txt")


def get_baseline_subreddits() -> List[str]:
    """
    Get a list of our baseline subreddits from a file
    """
    return read_file_by_lines("data/baseline_subreddits.txt")


def _get_month_from_json_filename(json_filename: str) -> str:
    """
    Get the month represented by a JSON filename as a YYYYMM string
    :param json_filename: Filename with the format YYYYMM.jsonl
    """
    return re.match(r"([0-9]{6}).jsonl", json_filename).group(1)


def get_subreddit_users(subreddit_name: str, start_timestamp: Union[datetime.datetime, int, None] = None,
                        end_timestamp: Union[datetime.datetime, int, None] = None, post_type: PostType = PostType.ALL) \
        -> Set[str]:
    """
    Get list of users from a subreddit
    :param subreddit_name: Name of the subreddit
    :param start_timestamp: Timestamp to get users after, can be None, int, or datetime object
    :param end_timestamp: Timestamp to get users before, can be None, int, or datetime object
    :param post_type: Type of post (all, post (submission), comment)
    """
    all_authors = set()

    jsons = get_subreddit_json(subreddit_name, post_type, properties=["author"], start_time=start_timestamp,
                               end_time=end_timestamp)
    for json_obj in jsons:
        all_authors.add(json_obj["author"])

    return all_authors


def get_posts_by_subreddit(subreddits: List[str], post_type: PostType, properties: Optional[List[str]] = None,
                           remap_properties: Optional[Dict[str, str]] = None,
                           start_time: Union[datetime.datetime, int, None] = None,
                           end_time: Union[datetime.datetime, int, None] = None, desc: bool = False,
                           filter_dates_strict: bool = False, ignore_automoderator: bool = False) \
        -> Dict[str, List[dict]]:
    """
    Get data from all subreddits in subreddits list. Returns a dictionary mapping subreddit name to a list
    of the data for that subreddit.
    :param subreddits: Name of the subreddits to get data from
    :param post_type: Type of post (all, post (submission), comment)
    :param properties: List of properties to fetch, for example "author" or "body" (default is fetch all)
    :param remap_properties: Properties to remap to a new name (like "author" to "user")
    :param start_time: Timestamp to get data after, can be None, int, or datetime object.
    :param end_time: Timestamp to get data before, can be None, int, or datetime object
    :param desc: Reverse order
    :param filter_dates_strict: if True, make sure post doesn't just come from subreddit file of the correct month
                                but verify that it's 'created_utc' actually falls within the given interval
    :param ignore_automoderator: Ignore the automoderator
    """
    data_by_subreddit = {}
    for subreddit in subreddits:
        # add 'created_utc' if it's not there so can do below filtering step
        if filter_dates_strict and 'created_utc' not in properties:
            properties.append('created_utc')
        post_list = list(get_subreddit_json(subreddit, post_type, properties=properties,
                                            remap_properties=remap_properties, start_time=start_time,
                                            end_time=end_time, desc=desc, ignore_automoderator=ignore_automoderator))
        # since get_subreddit_json rounds start & end times to nearest months, filter out posts that don't
        # actually fall within the given range
        if filter_dates_strict:
            post_list = [post for post in post_list if post["created_utc"] is not None
                         and start_time <= post["created_utc"] < end_time]
        data_by_subreddit[subreddit] = post_list
    return data_by_subreddit


def get_subreddit_json(subreddit_name: str, post_type: PostType, properties: Optional[List[str]] = None,
                       remap_properties: Optional[Dict[str, str]] = None,
                       start_time: Union[datetime.datetime, int, None] = None,
                       end_time: Union[datetime.datetime, int, None] = None, desc: bool = False,
                       ignore_automoderator: bool = False) -> Iterator[dict]:
    """
    Get data from a subreddit. Returns an iterator, use list(get_subreddit_json(...)) to get a list
    :param subreddit_name: Name of the subreddit
    :param post_type: Type of post (all, post (submission), comment)
    :param properties: List of properties to fetch, for example "author" or "body" (default is fetch all)
    :param remap_properties: Properties to remap to a new name (like "author" to "user")
    :param start_time: Timestamp to get data after, can be None, int, or datetime object
    :param end_time: Timestamp to get data before, can be None, int, or datetime object
    :param desc: Reverse order
    :param ignore_automoderator: Ignore the automoderator
    """
    return get_entity_json(EntityType.SUBREDDIT, subreddit_name, post_type, properties=properties,
                           remap_properties=remap_properties, start_time=start_time,
                           end_time=end_time, desc=desc, ignore_automoderator=ignore_automoderator)


def get_user_json(user_name: str, post_type: PostType, properties: List[str] = None,
                  remap_properties: Dict[str, str] = None, start_time: Union[datetime.datetime, int, None] = None,
                  end_time: Union[datetime.datetime, int, None] = None, desc: bool = False) -> Iterator[dict]:
    """
    Get data from a user. Returns an iterator, use list(get_user_json(...)) to get a list
    :param user_name: Name of the user
    :param post_type: Type of post (all, post (submission), comment)
    :param properties: List of properties to fetch, for example "author" or "body" (default is fetch all)
    :param remap_properties: Properties to remap to a new name (like "author" to "user")
    :param start_time: Timestamp to get data after, can be None, int, or datetime object
    :param end_time: Timestamp to get data before, can be None, int, or datetime object
    :param desc: Reverse order
    """
    return get_entity_json(EntityType.USER, user_name, post_type, properties=properties,
                           remap_properties=remap_properties, start_time=start_time,
                           end_time=end_time, desc=desc)


def get_entity_json(entity_type: EntityType, entity_name: str, post_type: PostType,
                    properties: Optional[List[str]] = None, remap_properties: Optional[Dict[str, str]] = None,
                    start_time: Union[datetime.datetime, int, None] = None,
                    end_time: Union[datetime.datetime, int, None] = None, desc: bool = False,
                    ignore_automoderator: bool = False) -> Iterator[dict]:
    """
    Same as get_subreddit_json and get_user_json, helper function that both use
    Calls _get_entity_json, which requires timestamps for dates and can't handle both posts and comments
    (this joins them together)
    """
    # allow both post types to be passed in
    if post_type == PostType.ALL:
        all_results = []
        for ind_post_type in PostType:
            if ind_post_type != PostType.ALL:
                all_results = itertools.chain(all_results, get_entity_json(
                    entity_type, entity_name, ind_post_type, properties=properties, remap_properties=remap_properties,
                    start_time=start_time, end_time=end_time, desc=desc, ignore_automoderator=ignore_automoderator))
        return all_results

    # convert all time arguments to UTC timestamps if not None
    start_timestamp = get_utc_timestamp(start_time) if type(start_time) == datetime.datetime else start_time
    end_timestamp = get_utc_timestamp(end_time) if type(end_time) == datetime.datetime else end_time

    return itertools.chain.from_iterable(
        _get_entity_json(entity_type, entity_name, post_type, properties=properties, remap_properties=remap_properties,
                         start_timestamp=start_timestamp, end_timestamp=end_timestamp, desc=desc,
                         ignore_automoderator=ignore_automoderator))


def _get_entity_json(entity_type: EntityType, entity_name: str, post_type: PostType,
                     properties: Optional[List[str]] = None,
                     remap_properties: Dict[str, str] = None, start_timestamp: Union[int, None] = None,
                     end_timestamp: Union[int, None] = None, desc: bool = False, ignore_automoderator: bool = False) \
        -> List[dict]:
    """
    Helper function for get_entity_json
    """
    file_dir = _get_json_dir(entity_type, entity_name, post_type)
    sorted_files = sorted(os.listdir(file_dir), reverse=desc)
    properties = set(properties) if properties is not None else properties
    for filename in sorted_files:
        file_json = []
        file_start_timestamp, file_end_timestamp = get_month_timestamps(_get_month_from_json_filename(filename))
        if start_timestamp is not None and file_end_timestamp < start_timestamp:
            yield file_json
        elif end_timestamp is not None and file_start_timestamp > end_timestamp:
            yield file_json
        else:
            full_path = os.path.join(file_dir, filename)
            for json_obj in _read_reddit_json(entity_type, entity_name, full_path, post_type,
                                              start_timestamp=start_timestamp, end_timestamp=end_timestamp, desc=desc,
                                              properties=properties, filter_moderators=False,
                                              ignore_automoderator=ignore_automoderator):
                if remap_properties is not None:
                    remap_properties = remap_properties if remap_properties is not None else {}
                    json_obj = {remap_properties.get(k, k): v for k, v in json_obj.items()}
                file_json.append(json_obj)
            yield file_json


def get_month_timestamps(month_str: str) -> Tuple[int, int]:
    """
    :param month_str: string representing the month, i.e. 202002
    :return: start and end timestamps for that month, INCLUSIVE for each
    """
    start_date = datetime.datetime.strptime(month_str, "%Y%m")
    end_date = start_date + relativedelta(months=1)
    return get_utc_timestamp(start_date), get_utc_timestamp(end_date) - 1


def get_utc_timestamp(datetime_obj: datetime.datetime) -> int:
    """
    Convert a datetime object to a UTC timestamp
    """
    return int(datetime_obj.replace(tzinfo=datetime.timezone.utc).timestamp())


def _get_json_dir(entity_type: EntityType, entity_name: str, post_type: PostType) -> str:
    """
    Get the appropriate directory to read JSON data
    :param entity_type: The type of the entity (subreddit or user)
    :param entity_name: The name of the entity (entity = subreddit or user)
    :param post_type: The type of post - post (submission) or comment
    """
    entity_type_dir = config.SUBREDDITS_DIR if entity_type == EntityType.SUBREDDIT else config.USERS_DIR
    post_type_dir = "comments" if post_type == PostType.COMMENT else "posts"
    return os.path.join(entity_type_dir, entity_name.lower(), post_type_dir, "json")


def _read_subreddit_moderators(subreddit_name: str) -> Set[str]:
    """
    Read list of moderators from a subreddit
    """
    moderator_file_path = os.path.join(config.SUBREDDITS_DIR, subreddit_name.lower(), config.MODERATORS_FILE_NAME)
    return set(read_file_by_lines(moderator_file_path))


def _read_reddit_json(entity_type: EntityType, entity_name: str, json_path: str, post_type: PostType,
                      filter_removed: bool = True, filter_moderators: bool = False,
                      start_timestamp: Union[int, None] = None, end_timestamp: Union[int, None] = None,
                      ignore_no_subreddit: bool = True, desc: bool = False, properties: Optional[Set[str]] = None,
                      ignore_automoderator: bool = False) \
        -> List[dict]:
    """
    Read JSON from one file and perform some filtering
    :param entity_type: The type of the entity (subreddit or user)
    :param entity_name: The name of the entity (entity = subreddit or user)
    :param json_path: The json path for the file to be read
    :param post_type: The type of post to return - post (submission) or comment
    :param filter_removed: Filter out posts that have been removed (default True)
    :param filter_moderators: Filter out known moderators (default False)
    :param start_timestamp: Don't return posts before this timestamp (default None)
    :param end_timestamp: Don't return posts after this timestamp (default None)
    :param ignore_no_subreddit: Ignore posts with no subreddit in the JSON object (this is rare but sometimes happens)
    :param desc: Reverse order of the posts
    :param properties: List of properties to fetch, for example "author" or "body" (default is fetch all)
    :param ignore_automoderator: Don't return posts by the automoderator
    """
    moderators = _read_subreddit_moderators(entity_name) if entity_type == EntityType.SUBREDDIT else set()
    with open(json_path, "r") as f:
        json_list = []
        for json_line in f:
            json_obj = json.loads(json_line.strip())
            if _should_return(json_obj, entity_type, post_type, moderators, filter_removed, filter_moderators,
                              start_timestamp, end_timestamp, ignore_no_subreddit, ignore_automoderator):
                if properties is not None:
                    json_obj = {k: v for k, v in json_obj.items() if k in properties}
                json_list.append(json_obj)
    if desc:
        json_list.reverse()
    return json_list


def _should_return(json_obj: dict, entity_type: EntityType, post_type: PostType, moderators: Set[str],
                   filter_removed: bool, filter_moderators: bool, start_timestamp: Union[int, None],
                   end_timestamp: Union[int, None], ignore_no_subreddit: bool, ignore_automoderator: bool):
    """
    Determines if json should be filtered out
    """
    if filter_moderators and entity_type == EntityType.SUBREDDIT and json_obj["author"] in moderators:
        # only remove moderators for subreddit, otherwise a user is being explicitly asked for
        return False
    if ignore_no_subreddit and "subreddit" not in json_obj:
        # remove json without a subreddit. Not really sure why this happens
        return False
    if start_timestamp is not None and json_obj["created_utc"] < start_timestamp:
        # filter posts before the start timestamp
        return False
    if end_timestamp is not None and json_obj["created_utc"] >= end_timestamp:
        # filter posts on or after the end timestamp
        return False
    if filter_removed:
        if post_type == PostType.POST:
            if is_removed_data_post_json(json_obj):
                return False
        else:
            if _is_removed_data_comment_json(json_obj):
                return False
    if ignore_automoderator and json_obj["author"].lower() == "automoderator":
        return False
    return True


def is_removed_data_post_json(json_obj: dict) -> bool:
    """
    Filter out deleted:
    1. Author is "[deleted]"
    2. Selftext is "[deleted]" or "[removed]"
    """
    if "selftext" not in json_obj:
        json_obj["selftext"] = ""
    return json_obj["author"] == "[deleted]" or json_obj["selftext"] in ["[deleted]", "[removed]"]


def _is_removed_data_comment_json(json_obj: dict) -> bool:
    """
    Filter out deleted:
    1. Author is "[deleted]"
    2. Body is "[deleted]" or "[removed]"
    """
    if "body" not in json_obj:
        json_obj["body"] = ""
    return json_obj["author"] == "[deleted]" or json_obj["body"] in ["[deleted]", "[removed]"]


def get_all_scraped_users() -> List[str]:
    """
    Get list of users that have been scraped
    """
    return os.listdir(config.USERS_DIR)


def get_all_scraped_subreddits() -> List[str]:
    """
    Get list of subreddits that have been scraped
    """
    return os.listdir(config.SUBREDDITS_DIR)


def get_text(post: dict, include_title: bool = False) -> str:
    """
    Get text from Reddit post JSON object
    """
    if include_title and "title" in post and "selftext" in post:
        return f"{post['title']}\n{post['selftext']}"
    elif include_title and "title" in post:
        return post["title"]
    elif "selftext" in post:
        return post["selftext"]
    elif "body" in post:
        return post["body"]
    else:
        return ""


def create_datetime_utc(y: int, m: int, d: int) -> datetime.datetime:
    return datetime.datetime(y, m, d).replace(tzinfo=datetime.timezone.utc)


def date_str_dash_to_obj(date_str: str) -> datetime.datetime.date:
    year, month, day = date_str.split('-')
    return datetime.datetime.date(datetime.datetime(int(year), int(month), int(day)))


def date_str_slash_to_obj(date_str: str) -> datetime.datetime.date:
    month, day, year = date_str.split('/')
    return datetime.datetime.date(datetime.datetime(int(year), int(month), int(day)))


def read_subreddit_timeseries(timeseries_path: str, subreddit_name: str, post_type: str):
    """
    Read existing timeseries for a subreddit
    """
    file_fmt = os.path.join(timeseries_path, f"{subreddit_name.lower()}_{post_type}.csv")
    return pd.read_csv(file_fmt, index_col=[0], parse_dates=[0]).sort_index()
