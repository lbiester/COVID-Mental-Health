"""
Create graph of Reddit users, with edges representing interactions (i.e. replies to posts)
"""
import argparse
import os
import pickle
import time
from datetime import datetime
from typing import List, Dict, Tuple, Set, Optional
from dateutil.relativedelta import *

import networkx as nx
import pandas as pd

from src.enums import PostType
from src.data_utils import get_utc_timestamp, get_posts_by_subreddit, read_file_by_lines
from src.graph.graph_utils import compute_graph_properties_undirected, compute_graph_properties_directed


"""
Notes: 

It looks like some prior work (e.g. Detecting Depression via Social Media) has computed daily measures of
social interactions/user graphs. So for now, the plan is to compute user graph properties for each day.
However, we also could analyze graphs from all posts in a week, month, etc.

First date with data appears to be 2008-12-16.
"""


class UserGraph:
    """
    Class to create graph representing user interactions on Reddit.
    """
    def __init__(self, start_dt: datetime, end_dt: datetime, subreddits: List[str], post2author: Dict[str, str],
                 verbose: Optional[bool] = False, compute_directed: bool = False):
        self.start_time = get_utc_timestamp(start_dt)
        self.end_time = get_utc_timestamp(end_dt)
        self.subreddits = subreddits
        self.post2author = post2author
        self.verbose = verbose
        self.post_dict, self.comment_dict, self.users = self._get_post_data()
        self.graph = nx.DiGraph()
        # add nodes (users) to graph based on who posted in the given time period
        self.graph.add_nodes_from(self.users)
        # add edges based on comments made in the given time period (this may create new nodes because comments may
        # be made on old posts)
        self._add_edges()
        self.graph_property_dict = None
        self.undirected_graph_property_dict = None
        self.compute_directed = compute_directed

    def compute_graph_properties(self) -> None:
        if self.compute_directed:
            if self.graph_property_dict is not None:
                print("Graph property dict already exists. Recomputing...")
            self.graph_property_dict = compute_graph_properties_directed(self.graph)
        else:
            self.graph_property_dict = {}
        undirected_graph = self.graph.to_undirected()
        if self.undirected_graph_property_dict is not None:
            print("Undirected property dict already exists. Recomputing...")
        self.undirected_graph_property_dict = compute_graph_properties_undirected(undirected_graph)

    def save(self, output_dir: str, graph_name: str) -> None:
        """
        Save graph data: self.graph, self.graph_property_dict, and self.graph_undirected_property_dict
        :param output_dir: directory to save data to
        :param graph_name: graph name to use in naming files
        """
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        nx.write_gpickle(self.graph, os.path.join(output_dir, "{}_graph.pkl".format(graph_name)))
        with open(os.path.join(output_dir, "{}_graph_property_dict.pkl".format(graph_name))) as f:
            pickle.dump(self.graph_property_dict, f)
        with open(os.path.join(output_dir, "{}_undirected_graph_property_dict.pkl".format(graph_name))) as f:
            pickle.dump(self.undirected_graph_property_dict, f)
        # write info about graphs --> start date, end date, subreddits used
        with open(os.path.join(output_dir, "{}_graph_info.txt".format(graph_name))) as f:
            f.write("start time: {}\n".format(self.start_time))
            f.write("end time: {}\n".format(self.end_time))
        with open(os.path.join(output_dir, "{}_subreddits.txt".format(graph_name)), "w+") as f:
            for subreddit in self.subreddits:
                f.write(subreddit+"\n")

    def _get_post_data(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]], Set[str]]:
        users = set()
        posts_by_subreddit = get_posts_by_subreddit(self.subreddits, PostType.POST, properties=["author", "id"],
                                                    start_time=self.start_time, end_time=self.end_time,
                                                    filter_dates_strict=True, filter_removed=False)
        comments_by_subreddit = get_posts_by_subreddit(self.subreddits, PostType.COMMENT,
                                                       properties=["author", "id", "parent_id", "link_id"],
                                                       start_time=self.start_time, end_time=self.end_time,
                                                       filter_dates_strict=True, ignore_automoderator=True,
                                                       filter_removed=False)
        # posts: map post id to author, subreddit, and post types
        post_dict = dict()
        for sr, post_list in posts_by_subreddit.items():
            for post in post_list:
                if post["author"] != "[deleted]":
                    post_dict[post["id"]] = {"author": post["author"], "subreddit": sr}
                    users.add(post["author"])
        # add comments to post dict: map post id to author, subreddit, parent id, link id, and post type
        comment_dict = dict()
        for sr, comment_list in comments_by_subreddit.items():
            for comment in comment_list:
                if comment["author"] != "[deleted]":
                    comment_dict[comment["id"]] = {"author": comment["author"], "subreddit": sr,
                                                   "parent_id": comment["parent_id"].split("_")[1],
                                                   "link_id": comment["link_id"].split("_")[1]}
                    users.add(comment["author"])
        return post_dict, comment_dict, users

    def _add_edges(self) -> None:
        oob = 0
        for comment_id, comment in self.comment_dict.items():
            author = comment["author"]
            parent_id = comment["parent_id"]
            if parent_id in self.post2author.keys():
                parent_author = self.post2author[parent_id]
            else:
                oob += 1
                continue
            # create edge
            self.graph.add_edge(author, parent_author)
        total = len(self.comment_dict)
        if self.verbose:
            print("could not find parent author for {} edges out of {} total".format(oob, total))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddits", type=str, nargs="+", help="List of subreddits (if not using a file)")
    parser.add_argument("--subreddit_file", type=str, help="Path to file with list of subreddits")
    parser.add_argument("--start_date", type=str,
                        help="Start date for interactions considered. Should be in MM/DD/YYYY format.")
    parser.add_argument("--end_date", type=str,
                        help="End date for interactions considered. Should be in MM/DD/YYYY format.")
    parser.add_argument("--interval_length", type=str, default="day",
                        help="Time period length to consider when creating a single graph. Options: day, month, year.")
    parser.add_argument("--output_dir", type=str, help="Path to directory to save data.")
    parser.add_argument("--graph_name", type=str, help="Graph name to use in naming saved data.")
    parser.add_argument("--num_groups", type=int, default=10, help="Number of groups to break up feature"
                                                                   " computation into.")
    parser.add_argument("--run_group", type=int, default=None, help="A single group to run")
    parser.add_argument("--compute_directed", action="store_true", help="Compute the directed graph")
    args = parser.parse_args()
    if ((args.subreddits is not None and args.subreddit_file is not None)
            or (args.subreddits is None and args.subreddit_file is None)):
        parser.error("Must specify subreddits OR subreddit_file")
    return args


def _datetime_str_to_obj(datetime_str: str) -> datetime:
    return datetime.strptime(datetime_str, "%m/%d/%Y")


def get_post_2_author(subreddits):
    # get mapping from post id to author for all posts and comments made on any of the subreddits
    # don't restrict by date for now because want to get info for posts that were created at a date outside of the
    # time range being considered BUT were commented on during the time range in consideration
    posts_by_subreddit = get_posts_by_subreddit(subreddits, PostType.ALL, properties=["author", "id"])
    # posts: map post id to author
    post2author = {post["id"]: post["author"] for sr, posts in posts_by_subreddit.items() for post in posts}
    empty_ids = "" in post2author.keys()
    none_ids = None in post2author.keys()
    if empty_ids:
        print("empty id found")
    if none_ids:
        print("none id found")
    empty_authors = "" in post2author.values()
    none_authors = None in post2author.values()
    if empty_authors:
        print("empty author found")
    if none_authors:
        print("none author found")
    return post2author


def get_time_intervals(start_dt: datetime, end_dt: datetime, interval_len: str) -> List[Tuple[str, datetime, datetime]]:
    intervals = []
    if interval_len == "day":
        interval_start = start_dt
        while interval_start <= end_dt:
            interval_end = interval_start + relativedelta(days=+1)
            intervals.append(("{}-{}-{}".format(interval_start.year, interval_start.month, interval_start.day),
                              interval_start, interval_end))
            interval_start = interval_start + relativedelta(days=+1)
    elif interval_len == "month":
        interval_start = start_dt
        while interval_start < end_dt:
            interval_end = interval_start + relativedelta(months=+1, days=-1)
            intervals.append(("{}_{}".format(interval_start.month, interval_start.year), interval_start, interval_end))
            interval_start = interval_end + relativedelta(days=+1)
    elif interval_len == "year":
        interval_start = start_dt
        while interval_start < end_dt:
            interval_end = interval_start + relativedelta(years=+1, days=-1)
            intervals.append(("{}".format(interval_start.year), interval_start, interval_end))
            interval_start = interval_end + relativedelta(days=+1)
    else:
        print("Invalid interval length {} given. Exiting.".format(interval_len))
        exit(1)
    return intervals


def save_graph_property_list(graph_property_list: List[dict], output_dir: str, group_num: int, graph_name: str,
                             is_final: bool = False):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "group_{}_{}.pkl".format(group_num, graph_name))
    if is_final:
        graph_property_df = pd.DataFrame(graph_property_list)
        graph_property_df.to_pickle(file_path)
    else:
        with open("{}.tmp".format(file_path), "wb") as f:
            pickle.dump(graph_property_list, f)


def run_group(subreddits: List[str], post2author: Dict[str, str], group_num: int, group_size: int, output_dir: str,
              intervals: List[Tuple[str, datetime, datetime]], graph_name: str, num_groups: int, interval_length: str,
              compute_directed: bool):
    output_file = os.path.join(output_dir, "group_{}_".format(group_num) + graph_name + ".pkl")
    if os.path.exists(output_file):
        # allows us to resume when script died in the middle for whatever reason
        # MAKE SURE THAT THE SAME NUMBER OF GROUPS ARE USED!
        return
    if os.path.exists("{}.tmp".format(output_file)):
        with open("{}.tmp".format(output_file), "rb") as f:
            graph_property_list = pickle.load(f)
        processed_intervals = {interval["interval_id"] for interval in graph_property_list}
    else:
        graph_property_list = []
        processed_intervals = set()
    print("working on group {}".format(group_num))
    start_idx = group_num * group_size
    end_idx = start_idx + group_size
    if group_num == num_groups - 1:
        end_idx = len(intervals)
    group_intervals = intervals[start_idx:end_idx]
    for name, start, end in group_intervals:
        if name in processed_intervals:
            print("Interval {} already processed, skipping".format(name))
            continue
        interval_dict = {"interval_id": name, "interval_start": start, "interval_end": end}
        # init graph
        init_time = time.time()
        user_graph = UserGraph(start, end, subreddits, post2author, verbose=True, compute_directed=compute_directed)
        # if there are no comments during the given interval, don't include the graph
        if not len(user_graph.comment_dict):
            print("WARNING: skipping interval {}, found no comments".format(name))
            continue
        user_graph.compute_graph_properties()
        if interval_length != "day":
            user_graph.save(output_dir, name + "_" + graph_name)
        print("created graph for interval ({}, {}) in time {}".format(start, end, time.time() - init_time))
        interval_dict.update(user_graph.graph_property_dict)
        # append 'u' to the front of undirected graph property dict names to distinguish from directed property dict
        # names
        u_dict = {"u_" + key: val for key, val in user_graph.undirected_graph_property_dict.items()}
        interval_dict.update(u_dict)
        graph_property_list.append(interval_dict)
        save_graph_property_list(graph_property_list, output_dir, group_num, graph_name, is_final=False)
    save_graph_property_list(graph_property_list, output_dir, group_num, graph_name, is_final=True)


def main() -> None:
    """
    Create user graphs based on user interactions.
    :return: None
    """
    # process arguments
    args = _parse_args()
    if args.subreddit_file:
        subreddits = read_file_by_lines(args.subreddit_file)
    else:
        subreddits = args.subreddits
    # covert dates to datetime objects
    start_dt = _datetime_str_to_obj(args.start_date)
    end_dt = _datetime_str_to_obj(args.end_date)
    # get post to author mapping
    init_time = time.time()
    post2author = get_post_2_author(subreddits)
    print("collected post 2 author in time {}".format(time.time() - init_time))
    # get time intervals to create graphs for
    intervals = get_time_intervals(start_dt, end_dt, args.interval_length)

    # break intervals up into groups
    num_groups = args.num_groups
    group_size = int(len(intervals) / num_groups)
    if args.run_group is None:
        for group_num in range(num_groups):
            run_group(subreddits, post2author, group_num, group_size, args.output_dir, intervals, args.graph_name,
                      num_groups, args.interval_length, args.compute_directed)
    else:
        run_group(subreddits, post2author, args.run_group, group_size, args.output_dir, intervals, args.graph_name,
                  num_groups, args.interval_length, args.compute_directed)


if __name__ == "__main__":
    main()
