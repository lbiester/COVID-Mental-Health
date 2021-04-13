"""
Create corpus of documents from Reddit data.
"""
import argparse
import os
import pickle
import random
import time
from copy import deepcopy
from datetime import datetime
from typing import List, Tuple, Dict, Any

import gensim.corpora as corpora
from argparse_utils import enum_action

from src.enums import PostType
from src.data_utils import get_utc_timestamp, get_posts_by_subreddit, read_file_by_lines
from src.text_utils import remove_special_chars_docs, split_into_sentences_docs, clean_and_tokenize_docs, \
    build_bigram_model, make_bigrams_docs, lemmatize_docs


class Corpus:
    """
    Class to read Reddit post/comment data and form corpus of document text, document ids, and vocabulary.
    """
    def __init__(self, start_dt: datetime, end_dt: datetime, subreddits: List[str], corpus_name: str,
                 downsample: bool = True, post_type: PostType = PostType.POST):
        self.start_time = get_utc_timestamp(start_dt)
        self.end_time = get_utc_timestamp(end_dt)
        self.subreddits = subreddits
        self.corpus_name = corpus_name
        self.post_type = post_type
        self.posts_by_subreddit = None
        if self.post_type in [PostType.ALL, PostType.POST]:
            self.posts_by_subreddit = get_posts_by_subreddit(self.subreddits, PostType.POST,
                                                             properties=["selftext", "title", "author",
                                                                         "created_utc", "id"],
                                                             start_time=self.start_time, end_time=self.end_time)
        self.comments_by_subreddit = None
        if self.post_type in [PostType.ALL, PostType.COMMENT]:
            self.comments_by_subreddit = get_posts_by_subreddit(self.subreddits, PostType.COMMENT,
                                                                properties=["body", "author", "created_utc", "id"],
                                                                start_time=self.start_time, end_time=self.end_time)
        if downsample:
            self.post_dict_by_date = self._collect_posts_by_date_and_type()
            self.id2text = self._get_id2text()
            self.doc_text_list, self.doc_id_list = self._select_docs()
        else:
            # get doc_text and doc_id lists from all posts + comments
            self.doc_text_list, self.doc_id_list = self._collect_all_docs()
        print("corpus size is {}".format(len(self.doc_text_list)))
        self.vocab_dict = None
        self.doc_tf_list = []

    def _collect_all_docs(self) -> Tuple[List[str], List[str]]:
        """
        Collect all posts and comments (no downsampling).
        """
        doc_text_list = []
        doc_id_list = []
        if self.posts_by_subreddit is not None:
            for post_list in self.posts_by_subreddit.values():
                for post in post_list:
                    doc_text_list.append(post["title"] + " " + post["selftext"])
                    doc_id_list.append(post['id'])
        if self.comments_by_subreddit is not None:
            for comment_list in self.comments_by_subreddit.values():
                for comment in comment_list:
                    doc_text_list.append(comment["body"])
                    doc_id_list.append(comment['id'])
        return doc_text_list, doc_id_list

    def _collect_posts_by_date_and_type(self):
        """
        Group posts by date, post_type (comment vs post), and subreddit
        """
        post_dict_by_date = dict()
        if self.posts_by_subreddit is not None:
            self._collect_posts_by_date(post_dict_by_date, self.posts_by_subreddit, PostType.POST)
        if self.comments_by_subreddit is not None:
            self._collect_posts_by_date(post_dict_by_date, self.comments_by_subreddit, PostType.COMMENT)
        return post_dict_by_date

    @staticmethod
    def _collect_posts_by_date(post_dict_by_date: Dict[datetime.date, Dict[PostType, Dict[str, List[str]]]],
                               posts_by_subreddit: Dict[str, List[Dict[str, Any]]], post_type: PostType) -> None:
        for sr, post_list in posts_by_subreddit.items():
            for post in post_list:
                date = datetime.utcfromtimestamp(post["created_utc"]).date()
                if date not in post_dict_by_date:
                    post_dict_by_date[date] = dict()
                if post_type not in post_dict_by_date[date]:
                    post_dict_by_date[date][post_type] = dict()
                if sr not in post_dict_by_date[date][post_type]:
                    post_dict_by_date[date][post_type][sr] = []
                post_dict_by_date[date][post_type][sr].append(post["id"])

    def _get_id2text(self) -> Dict[str, str]:
        id2text = dict()
        # add posts
        if self.posts_by_subreddit is not None:
            for post_list in self.posts_by_subreddit.values():
                for post in post_list:
                    # add title to post text
                    id2text[post["id"]] = post["title"] + " " + post["selftext"]
        # add comments
        if self.comments_by_subreddit is not None:
            for comment_list in self.comments_by_subreddit.values():
                for comment in comment_list:
                    id2text[comment["id"]] = comment["body"]
        return id2text

    def _select_docs(self) -> Tuple[List[str], List[str]]:
        """
        Select documents to include in corpus such that there are are an equal number of posts for each day,
        and for each day, there are equal number of posts by post type (comment vs post) and subreddit.
        """
        # get min count of posts for a single day, post type, and subreddit combination
        post_counts = [len(post_list) for post_dict in self.post_dict_by_date.values() for sr_dict in post_dict.values()
                       for post_list in sr_dict.values()]
        select_count = min(post_counts)
        print("min number of posts across all date, type, sr combinations is {}".format(select_count))
        doc_id_list = []
        for post_dict in self.post_dict_by_date.values():
            for sr_dict in post_dict.values():
                for post_list in sr_dict.values():
                    # select select_count number of posts
                    selected_ids = random.sample(post_list, select_count)
                    doc_id_list += selected_ids
        doc_text_list = [self.id2text[post_id] for post_id in doc_id_list]
        return doc_text_list, doc_id_list

    def _create_metadata_dict(self) -> Dict[str, Dict[str, str]]:
        metadata_dict = {}
        for subreddit, post_list in self.posts_by_subreddit.items():
            for post_dict in post_list:
                post_id = post_dict["id"]
                post_md_dict = deepcopy(post_dict)
                post_md_dict.pop('selftext', None)
                post_md_dict.pop('title', None)
                post_md_dict.pop('body', None)
                post_md_dict['subreddit'] = subreddit
                metadata_dict[post_id] = post_md_dict
        return metadata_dict

    def prepocess_posts(self):
        # clean, tokenize, add bigrams, lemmatize
        self._clean_and_tokenize()
        self._add_bigrams()
        self._remove_sentence_boundaries()
        self._lemmatize()

    def _clean_and_tokenize(self):
        # clean, tokenize, and remove stopwords
        # also transform each doc to list of sentences
        self.doc_text_list = remove_special_chars_docs(self.doc_text_list)
        self.doc_text_list = split_into_sentences_docs(self.doc_text_list)
        self.doc_text_list = clean_and_tokenize_docs(self.doc_text_list)

    def _add_bigrams(self):
        # build bigram model
        sentence_list = [sent for doc in self.doc_text_list for sent in doc]
        bigram_model = build_bigram_model(sentence_list)
        # identify bigram phrases and convert them to this
        self.doc_text_list = make_bigrams_docs(self.doc_text_list, bigram_model)

    def _remove_sentence_boundaries(self):
        # convert each document from list of sentences to list of words
        self.doc_text_list = [[word for sent in doc for word in sent] for doc in self.doc_text_list]

    def _lemmatize(self):
        # lemmatize words in each document
        self.doc_text_list = lemmatize_docs(self.doc_text_list)

    def remove_empty_posts(self):
        keep_idxs = set([idx for idx, doc in enumerate(self.doc_text_list) if doc])
        self.doc_text_list = [doc for idx, doc in enumerate(self.doc_text_list) if idx in keep_idxs]
        self.doc_id_list = [id for idx, id in enumerate(self.doc_id_list) if idx in keep_idxs]

    def create_vocab_dict(self, no_below: int = 25, no_above: float = .5):
        """
        Create vocab dictionary (word id -> word) from all documents.
        :param no_below: remove words that have less than this many occurrences across all documents
        :param no_above: remove words that appear in more than this % of documents
        """
        if self.vocab_dict is not None:
            print("Already created vocab dict. Recreating with current document list.")
        self.vocab_dict = corpora.Dictionary(self.doc_text_list)
        self.vocab_dict.filter_extremes(no_below=no_below, no_above=no_above)

    def load_vocab_dict(self, vocab_path: str) -> None:
        """
        Load vocab dict from file.
        :param vocab_path: path to file containing vocab dict
        """
        self.vocab_dict = corpora.dictionary.Dictionary.load(vocab_path)

    def create_doc_tf_list(self):
        # create list of each document where each document is a list of the term frequencies of its words
        if len(self.doc_tf_list):
            print("Document term frequency list was already created. Recreating with current vocab dict and doc list.")
        self.doc_tf_list = [self.vocab_dict.doc2bow(doc) for doc in self.doc_text_list]

    def save(self, output_dir: str):
        """
        Save corpus data (doc_text_list, doc_id_list, vocab_dict, doc_tf_list, and metadata_dict) to files.
        Also save start date, end date and post types to file.
        :param output_dir: directory to save corpus data to
        """
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        with open(os.path.join(output_dir, "{}_doc_texts.pkl".format(self.corpus_name)), "wb") as f:
            pickle.dump(self.doc_text_list, f)
        with open(os.path.join(output_dir, "{}_doc_ids.pkl".format(self.corpus_name)), "wb") as f:
            pickle.dump(self.doc_id_list, f)
        self.vocab_dict.save(os.path.join(output_dir, "{}_vocab.dct".format(self.corpus_name)))
        with open(os.path.join(output_dir, "{}_doc_tfs.pkl".format(self.corpus_name)), "wb") as f:
            pickle.dump(self.doc_tf_list, f)
        with open(os.path.join(output_dir, "{}_corpus_info.txt".format(self.corpus_name)), "w+") as f:
            f.write("start time: {}\n".format(self.start_time))
            f.write("end time: {}\n".format(self.end_time))
            f.write("post type: {}\n".format(self.post_type))
        with open(os.path.join(output_dir, "{}_subreddits.txt".format(self.corpus_name)), "w+") as f:
            for subreddit in self.subreddits:
                f.write(subreddit+"\n")


def load_corpus_data(input_dir: str, corpus_name: str) -> Tuple[List[str], List[str], List[List[Tuple[int, int]]],
                                                                corpora.Dictionary]:
    """
    Load corpus data needed for training topic model from files (doc_list, vocab_dict, and doc_tf_list).
    :param input_dir: directory to read corpus data from
    :param corpus_name: name of corpus to get data from
    :return doc_text_list: list of document strings
            doc_id_list: list of coument ids
            doc_tf_list: list of document term-frequency lists, where each tf-list is a list of (term id, tf) tuples
            vocab_dict: dictionary mapping word ids to words
    """
    with open(os.path.join(input_dir, "{}_doc_texts.pkl".format(corpus_name)), "rb") as f:
        doc_text_list = pickle.load(f)
    with open(os.path.join(input_dir, "{}_doc_ids.pkl".format(corpus_name)), "rb") as f:
        doc_id_list = pickle.load(f)
    vocab_dict = corpora.dictionary.Dictionary.load(os.path.join(input_dir, "{}_vocab.dct".format(corpus_name)))
    with open(os.path.join(input_dir, "{}_doc_tfs.pkl".format(corpus_name)), "rb") as f:
        doc_tf_list = pickle.load(f)
    return doc_text_list, doc_id_list, doc_tf_list, vocab_dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddits", type=str, nargs="+", help="List of subreddits (if not using a file)")
    parser.add_argument("--subreddit_file", type=str, help="Path to file with list of subreddits")
    parser.add_argument("--name", type=str, help="Name of corpus. Used in naming files when corpus data is saved.")
    parser.add_argument("--start_date", type=str, default="08/12/2019",
                        help="Start date for collecting posts. Should be in MM/DD/YYYY format.")
    parser.add_argument("--end_date", type=str, default="08/12/2019",
                        help="End date for collecting posts. Should be in MM/DD/YYYY format.")
    parser.add_argument("--output_dir", type=str, help="Path to directory to save data.")
    parser.add_argument("--vocab", type=str, help="Path to file with vocab dictionary to use when converting"
                                                  "docs to term-frequency lists. Will create new vocab using docs in"
                                                  "this corpus if existing vocab dict is not provided.")
    parser.add_argument("--downsample", action='store_true', help='If true, randomly downsample the data so that an '
                                                                  'even number of posts and comments from each '
                                                                  'subreddit are selected.')
    parser.add_argument("--post_type", action=enum_action(PostType), default=PostType.POST, help="Post type to process")
    args = parser.parse_args()
    if ((args.subreddits is not None and args.subreddit_file is not None)
            or (args.subreddits is None and args.subreddit_file is None)):
        parser.error("Must specify subreddits OR subreddit_file")
    return args


def _datetime_str_to_obj(datetime_str: str) -> datetime:
    return datetime.strptime(datetime_str, "%m/%d/%Y")


def main() -> None:
    """
    Collect corpus of documents from Reddit posts/comments.
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

    # create corpus
    init_time = time.time()
    corpus = Corpus(start_dt, end_dt, subreddits, args.name, downsample=args.downsample, post_type=args.post_type)
    create_time = time.time()
    print("Created corpus (posts by subreddit + doc list) in time {}".format(create_time - init_time))
    corpus.prepocess_posts()
    preprocess_time = time.time()
    print("Preprocessed posts in time {}".format(preprocess_time - create_time))
    corpus.remove_empty_posts()
    re_time = time.time()
    print("Removed empty posts in time {}".format(re_time - preprocess_time))
    if args.vocab is None:
        corpus.create_vocab_dict()
        vocab_time = time.time()
        print("Created vocab dict in time {}".format(vocab_time - re_time))
    else:
        corpus.load_vocab_dict(args.vocab)
        vocab_time = time.time()
        print("Loaded vocab dict in time {}".format(vocab_time - re_time))
    corpus.create_doc_tf_list()
    tf_time = time.time()
    print("Created doc tf list in time {}".format(tf_time - vocab_time))

    # save corpus
    corpus.save(args.output_dir)
    save_time = time.time()
    print("Saved Corpus in time {}".format(save_time - tf_time))


if __name__ == "__main__":
    main()
