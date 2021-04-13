import os
import pickle
import re
from collections import defaultdict
from typing import List, Dict, Tuple

import nltk
import numpy as np

from src.config import LIWC_2015_PATH
from src.text_utils import remove_special_chars


class LIWCProcessor:
    TOKEN_TO_IDX_PATH = f"{LIWC_2015_PATH}_token_to_idx.pickle"
    SAVE_TOKEN_TO_IDX_FREQUENCY = 1000
    SAVE_TOKEN_TO_IDX_NEW_TOKENS = 1000

    def __init__(self, ignore_case: bool = True, custom_tokenize: bool = True):
        self.ignore_case = ignore_case
        self.liwc_category_to_words = read_liwc_file()
        self.liwc_categories = sorted(self.liwc_category_to_words.keys())
        self.liwc_categories_to_idx = {category: i for i, category in enumerate(self.liwc_categories)}
        self.token_to_idx = self.load_token_to_idx()
        self.count = 0
        self.new_tokens = 0
        self.custom_tokenize = custom_tokenize

    def vectorize(self, text: str) -> np.array:
        features = np.zeros(len(self.liwc_categories) + 1, dtype=np.uint16)
        if self.ignore_case:
            text = text.lower()
        if self.custom_tokenize:
            tokenized_text = liwc_tokenize(text)
        else:
            tokenized_text = nltk.word_tokenize(text)
        for token in tokenized_text:
            if token in self.token_to_idx:
                if len(self.token_to_idx[token]) != 0:
                    features[self.token_to_idx[token]] += 1
            else:
                token_to_idx = set()
                for category, (full_patterns, sw_patterns) in self.liwc_category_to_words.items():
                    if token in full_patterns:
                        features[self.liwc_categories_to_idx[category]] += 1
                        token_to_idx.add(self.liwc_categories_to_idx[category])
                    elif token.startswith(sw_patterns):
                        features[self.liwc_categories_to_idx[category]] += 1
                        token_to_idx.add(self.liwc_categories_to_idx[category])
                self.token_to_idx[token] = np.array(sorted(token_to_idx))
                self.new_tokens += 1
        self.save_token_to_idx()
        # final feature is the number of total tokens
        features[-1] = len(tokenized_text)
        return features

    def load_token_to_idx(self) -> Dict[str, Tuple[int]]:
        try:
            if os.path.isfile(self.TOKEN_TO_IDX_PATH):
                with open(self.TOKEN_TO_IDX_PATH, "rb") as f:
                    categories, token_to_idx = pickle.load(f)
                    if categories == self.liwc_categories:
                        return token_to_idx
        except:
            return {}

    def save_token_to_idx(self):
        # only save token_to_idx if meeting two thresholds, for number of items processed and number of new tokens
        self.count += 1
        if self.count >= self.SAVE_TOKEN_TO_IDX_FREQUENCY and self.new_tokens >= self.SAVE_TOKEN_TO_IDX_NEW_TOKENS:
            with open(self.TOKEN_TO_IDX_PATH, "wb") as f:
                pickle.dump((self.liwc_categories, self.token_to_idx), f)
            self.new_tokens = 0
            self.count = 0


def clean_token(token: str) -> str:
    if token.isalnum():
        return token
    # remove non alphanum chars from beginning and end of str
    return re.sub("^\W*(.*?)\W*$", r"\1", token)


def liwc_tokenize(text: str) -> List[str]:
    # tokenize for input to LIWC
    # do not want to use nltk.word_tokenize because it

    # step 1: split on whitespace
    text_list = remove_special_chars(text).split()

    # step 2: remove punctuation at the end of each string
    text_list_cleaned = []
    for token in text_list:
        cleaned = clean_token(token)
        if len(cleaned) > 0:
            text_list_cleaned.append(cleaned)
    return text_list_cleaned


def read_liwc_file() -> Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    liwc_match_list = defaultdict(lambda: ([], []))
    with open(LIWC_2015_PATH) as f:
        for line in f.read().splitlines():
            pattern, category = (x.strip() for x in line.split(","))
            wildcard = pattern[-1] == "*"
            if wildcard:
                pattern = pattern[:-1]
                liwc_match_list[category][1].append(pattern)
            else:
                liwc_match_list[category][0].append(pattern)
    liwc_match_tuples = {category: (tuple(full_patterns), tuple(sw_patterns))
                         for category, (full_patterns, sw_patterns) in liwc_match_list.items()}
    return liwc_match_tuples
