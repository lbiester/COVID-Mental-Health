"""
Train topic models for analyzing Reddit post and comments content.
"""
import argparse
import os
import time
import pandas as pd
from typing import List, Tuple

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

from src.topics.corpus import load_corpus_data


class TopicModel:
    """
    Class to create LDA topic model.
    """
    def __init__(self, doc_list: List[str], doc_tf_list: List[List[Tuple[int, int]]], vocab_dict: corpora.Dictionary,
                 num_topics: int, mallet_tmp_dir: str, verbose: bool = False):
        self.doc_list = doc_list
        self.doc_tf_list = doc_tf_list
        self.vocab_dict = vocab_dict
        self.num_topics = num_topics
        self.mallet_tmp_dir = mallet_tmp_dir
        if not os.path.exists(mallet_tmp_dir):
            os.makedirs(mallet_tmp_dir)
        self.verbose = verbose
        self.model = None

    def train_lda_mallet_model(self, mallet_path: str, workers: int = 1):
        """
        Train Mallet LDA topic model (http://mallet.cs.umass.edu/index.php)
        :param mallet_path: path to mallet model (e.g. /home/katiemat/tools/mallet-2.0.8/bin/mallet)
        :param workers: number of threads to use in training model (-1 = all possible)
        """
        if self.model is not None:
            print("Existing model found. Deleting and retraining.")
        start_time = time.time()
        self.model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=self.doc_tf_list, num_topics=self.num_topics,
                                                      id2word=self.vocab_dict, workers=workers,
                                                      prefix=os.path.join(self.mallet_tmp_dir,
                                                                          '{}_'.format(self.num_topics)))
        train_time = time.time() - start_time
        if self.verbose:
            print("Finished training {} topic LDA mallet model in time {}".format(self.num_topics, train_time))

    def compute_model_coherence(self, coherence_type: str = 'c_v'):
        """
        Compute coherence score for topics produced by model.
        Scores are based on similiarity of top words within each topic.
        See https://radimrehurek.com/gensim/models/coherencemodel.html
        and http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf
        for more info on the difference types of coherence scores.
        """
        coherence_model = CoherenceModel(model=self.model, texts=self.doc_list, dictionary=self.vocab_dict,
                                         coherence=coherence_type)
        coherence = coherence_model.get_coherence()
        if self.verbose:
            print("Coherence score for {} topic model was {}".format(self.num_topics, coherence))
        return coherence

    def save_model(self, output_dir: str, model_name: str):
        """
        Save topic model to file in output_dir.
        """
        self.model.save(os.path.join(output_dir, "{}_{}_topics.mdl".format(model_name, self.num_topics)))


def sweep_num_topics(doc_list: List[str], doc_tf_list: List[List[Tuple[int, int]]], vocab_dict: corpora.Dictionary,
                     output_dir: str, model_base_name: str, mallet_path: str, mallet_tmp_dir: str, start: int = 10,
                     step: int = 5, limit: int = 41, workers: int = 1, verbose: bool = False) -> List[Tuple[int, float]]:
    """
    Train models with different ks and compute coherence scores.
    :param doc_list: list of document strings
    :param doc_tf_list: list of document term-frequency lists, where each tf-list is a list of (term id, tf) tuples
    :param vocab_dict: dictionary mapping word ids to words
    :param output_dir: directory to save models to
    :param model_base_name: name of topic model, used in naming saved files
    :param mallet_path: path to mallet model binary
    :param mallet_tmp_dir: path to save mallet 'temporary' files to (although they're needed later)
    :param start: starting k
    :param step: step size for k
    :param limit: limit on k values
    :param workers: number of threads to use in training each model
    :param verbose: If true, print training progress.
    :return coherence_scores: List of tuples: (k, coherence) for each model trained
    """
    coherence_scores = []
    for num_topics in range(start, limit, step):
        topic_model = TopicModel(doc_list, doc_tf_list, vocab_dict, num_topics, mallet_tmp_dir, verbose)
        topic_model.train_lda_mallet_model(mallet_path, workers=workers)
        topic_model.save_model(output_dir, model_base_name)
        coherence = topic_model.compute_model_coherence()
        coherence_scores.append((num_topics, coherence))
    return coherence_scores


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, help="Path to directory of corpus to use for training")
    parser.add_argument("--corpus_name", type=str,
                        help="Name of corpus to use. Used to read in correct files within corpus_dir.")
    parser.add_argument("--mallet_path", type=str, help="Path to mallet model binary.")
    parser.add_argument("--max_num_topics", type=int, default=40, help="Maximum number of topics to use when training model.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of threads ot use in training topic model.")
    parser.add_argument("--output_dir", type=str, help="Path to directory to save data.")
    parser.add_argument('--base_model_name', type=str, help="Base name of topic models to use when saving files.")
    parser.add_argument('--mallet_tmp_dir', type=str, help="Path to directory to save mallet 'temporary' files to"
                                                           "(note that these files are actually needed when using saved models)")
    args = parser.parse_args()
    return args


def main() -> None:
    """
    Train topics models, sweeping over k=number of topics.
    """
    # process arguments
    args = _parse_args()

    # load data
    doc_list, _, doc_tf_list, vocab_dict = load_corpus_data(args.corpus_dir, args.corpus_name)

    # train models
    coherence_scores = sweep_num_topics(doc_list, doc_tf_list, vocab_dict, args.output_dir, args.base_model_name,
                                        args.mallet_path, args.mallet_tmp_dir, limit=args.max_num_topics + 1,
                                        workers=args.num_workers, verbose=True)

    # save coherence scores in csv file
    coherence_df = pd.DataFrame(coherence_scores, columns=["num topics", "coherence score"])
    coherence_df.to_csv(os.path.join(args.output_dir, "{}_coherence_scores.csv".format(args.base_model_name)))


if __name__ == "__main__":
    main()
