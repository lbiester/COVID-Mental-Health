"""
Script for applying topic model to corpus of documents to get the distribution of topics within those documents.
"""
import argparse
import os
import pickle
from datetime import datetime
from typing import Tuple, List, Dict, Optional

import gensim
import gensim.corpora as corpora
import numpy as np
import pandas as pd

from src.data_utils import read_file_by_lines, post_type_str_to_enum, get_posts_by_subreddit
from src.enums import PostType
from src.topics.corpus import load_corpus_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic_model_path", type=str, help="Path to topic model")
    parser.add_argument("--corpus_dir", type=str, help="Path directory containing corpus of documents to run model on.")
    parser.add_argument("--corpus_name", type=str, help="Prefix used in naming corpus files.")
    parser.add_argument("--post_type", type=str, default="post", help="What types of documents are in your corpus."
                                                                       "Options are: all, post, comment.")
    parser.add_argument("--output_dir", type=str, help="Path to directory to save computed topic dist features.")
    parser.add_argument('--vocab_path', type=str, help="Path to vocab dictionary to use to map corpus texts to"
                                                       " term-frequency representation. If not provided, will use the"
                                                       " term-frequency representation created during corpus creation.")
    parser.add_argument("--mallet_path", type=str, help="Path to mallet binary - by default, will use whatever is"
                                                        " stored in the topic model file")
    parser.add_argument("--new_corpus_name", type=str, help="Alternative corpus name to use for output file")
    args = parser.parse_args()
    return args


def load_document_data(corpus_dir: str, corpus_name: str, post_type: PostType, vocab_path: Optional[str] = None) -> \
        Tuple[List[List[Tuple[int, int]]], Dict[int, datetime.date]]:
    if not vocab_path:
        # load corpus of documents and their TF representations
        doc_text_list, doc_id_list, doc_tf_list, vocab_dict = load_corpus_data(corpus_dir, corpus_name)
    else:
        # otherwise just load doc text and doc id list, and use vocab dict at vocab path to get doc_tf_list
        with open(os.path.join(corpus_dir, "{}_doc_texts.pkl".format(corpus_name)), "rb") as f:
            doc_text_list = pickle.load(f)
        with open(os.path.join(corpus_dir, "{}_doc_ids.pkl".format(corpus_name)), "rb") as f:
            doc_id_list = pickle.load(f)
        vocab_dict = corpora.dictionary.Dictionary.load(vocab_path)
        doc_tf_list = [vocab_dict.doc2bow(doc) for doc in doc_text_list]

    # load info associated with corpus creation (what posts were used)
    lines = read_file_by_lines(os.path.join(corpus_dir, "{}_corpus_info.txt".format(corpus_name)))
    start_time, end_time = lines[0], lines[1]
    start_time = datetime.utcfromtimestamp(int(start_time.split(' ')[2]))
    end_time = datetime.utcfromtimestamp(int(end_time.split(' ')[2]))
    subreddits = read_file_by_lines(os.path.join(corpus_dir, "{}_subreddits.txt".format(corpus_name)))

    # load metadata associated with each document
    # get metadata associated with documents in corpus
    posts_by_subreddit = get_posts_by_subreddit(subreddits, post_type, properties=["created_utc", "id"],
                                                start_time=start_time, end_time=end_time)
    # create dict mapping date to index in doc text list
    id2idx = {post_id: post_idx for post_idx, post_id in enumerate(doc_id_list)}
    idx2date = {}
    for post_list in posts_by_subreddit.values():
        for post in post_list:
            post_id = post["id"]
            post_date = datetime.utcfromtimestamp(post["created_utc"]).date()
            if post_id in id2idx:
                # post_id could be missing from corpus because did not have enough text to be valid document
                post_idx = id2idx[post_id]
                idx2date[post_idx] = post_date
    return doc_tf_list, idx2date


def get_doc_topic_metrics(doc_tf_list: List[List[Tuple[int, int]]], idx2date: Dict[int, datetime.date],
                          topic_model: gensim.models.wrappers.ldamallet.LdaMallet, output_dir: str,
                          corpus_name: str, post_type: str):
    # get topic distribution for each document
    doc_topic_list = topic_model[doc_tf_list]
    doc_topic_matrix = np.array(doc_topic_list)
    doc_topic_matrix = doc_topic_matrix[:, :, 1]
    # put into pandas dataframe
    # use np.Nan to indicate that doc idx is not in idx2date (this means associated doc is of the incorrect post type)
    index = [idx2date[idx] if idx in idx2date else np.NaN for idx in range(len(doc_tf_list))]
    doc_topic_df = pd.DataFrame(index=index, data=doc_topic_matrix,
                                columns=["topic_{}".format(topic_num)
                                         for topic_num in range(doc_topic_matrix.shape[1])])
    # drop entries with 'nan' index --> they are incorrect post type
    doc_topic_df['temp_col'] = doc_topic_df.index
    doc_topic_df.dropna(inplace=True)
    doc_topic_df.drop('temp_col', axis=1, inplace=True)
    # group by index value and take mean and then save
    doc_topic_df.index.name = 'index'
    doc_topic_df = doc_topic_df.groupby('index').mean()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    doc_topic_df.to_csv(os.path.join(output_dir, "{}_{}_topic_dist.csv".format(corpus_name, post_type)))


def main():
    args = _parse_args()
    post_type = post_type_str_to_enum(args.post_type)

    # load data associated with corpus of documents to get topics for
    doc_tf_list, idx2date = load_document_data(args.corpus_dir, args.corpus_name, post_type, args.vocab_path)

    # load topic model
    topic_model = gensim.models.wrappers.ldamallet.LdaMallet.load(args.topic_model_path)
    if args.mallet_path is not None:
        topic_model.mallet_path = args.mallet_path

    # compute and save metrics
    output_corpus_name = args.corpus_name if args.new_corpus_name is None else args.new_corpus_name
    get_doc_topic_metrics(doc_tf_list, idx2date, topic_model, args.output_dir, output_corpus_name, args.post_type)


if __name__ == "__main__":
    main()
