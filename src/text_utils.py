import re

from IPython import embed
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import nltk
import numpy as np

from src import config
from src.data_utils import read_file_by_lines


# prepare stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
# remove words in LIWC from stopwords list
liwc_lines = read_file_by_lines(config.LIWC_2015_PATH)
liwc_words = set([line.split(' ')[0] for line in liwc_lines])
liwc_star_words = set([lw for lw in liwc_words if lw[-1] == '*'])
sw = [word for word in sw if word not in liwc_words]
sw = [word for word in sw if not np.any([word.startswith(lw[:-1]) for lw in liwc_star_words])]
STOP_WORDS = sw


def lemmatize_docs(docs):
    """
    Lemmatize words in each doc. Each doc is a list of words.
    """
    nlp = spacy.load('en', disable=['parser', 'ner'])  # only keep tagger component for efficiency
    docs = [lemmatize(doc, nlp) for doc in docs]
    return docs


def lemmatize(doc, nlp):
    """
    Lemmatize words in doc.
    :param: doc: list of words (strs).
    :param: nlp: spacy nlp preprocessor
    """
    doc = nlp(" ".join(doc))
    doc = [word.lemma_ for word in doc]
    return doc


def make_bigrams_docs(docs, bigram_model):
    """
    Convert pairs of words that are bigram phrases (as identified by bigram model) into bigrams.
    :param: docs: list of  of docs, where each doc is list of sentences and each sentence is a list of words and each word is a str
    :param: bigram_model: gensim Phrases bigram model
    """
    return [make_bigrams(doc, bigram_model) for doc in docs]


def make_bigrams(sentences, bigram_model):
    """
    Convert pairs of words that are bigram phrases (as identified by bigram model) into bigrams.
    E.g. "oil" "leak" --> "oil_leak"
    :param: sentences: list of list of words (strs)
    :param: bigram_model: gensim Phrases bigram model
    """
    return [bigram_model[sent] for sent in sentences]


def build_bigram_model(sentences, min_count=5, threshold=100):
    """
    Build bigram model.
    :param: sentences: List[List[str]] List of sentences, where each sentence is a list of words.
    :param: min_count: ignore all words and bigrams with total collected count lower than this value
    :param: threshold: Threshold score for forming bigrams. Pairs of words are converted to bigrams if they have a score
    greater than the threshold. A higher score means fewer bigrams.
    """
    bigram = gensim.models.Phrases(sentences, min_count=min_count, threshold=threshold)
    # convert to faster implementation of Phrases (reduced functionality, but is all we need since we aren't going to
    # add more words to the models)
    bigram_model = gensim.models.phrases.Phraser(bigram)
    return bigram_model


def clean_and_tokenize_docs(docs):
    """
    Tokenizes each doc (list of sentence strs) in docs. Also removes punctuation and stopwords
    and converts to lower case.
    """
    return [clean_and_tokenize(doc) for doc in docs]


def clean_and_tokenize(sentences):
    """
    Converts each sentence (str) in sentences (list of strs) into a list of words.
    Also cleans text by removing punctuation, removing stopwords, and converting to lowercase.
    """
    # deacc=True to remove punctuation
    sentences = [[word for word in simple_preprocess(sent, deacc=True) if word not in STOP_WORDS] for sent in sentences]
    return sentences


def split_into_sentences_docs(docs):
    """
    Split each doc (str) in docs (list of strs) into a list of sentence strings.
    """
    nltk.download('punkt')
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return [split_into_sentences(doc, sent_tokenizer) for doc in docs]


def split_into_sentences(text, sent_tokenizer):
    """
    Split text (str) into list of sentences
    """
    return sent_tokenizer.tokenize(text)


def remove_special_chars_docs(docs):
    """
    Remove special characters from each doc (str) in a list of docs common in Reddit Posts.
    Also remove emails, URLS, and IP addresses.
    """
    return [remove_special_chars(doc) for doc in docs]


def remove_special_chars(text):
    """
    Remove special characters from text common in Reddit Posts as well as emails, URLS, and IP addresses.
    Adapted from https://github.com/LoLei/redditcleaner/blob/master/redditcleaner/__init__.py
    """
    # Newlines (replaced with space to preserve cases like word1\nword2)
    text = re.sub(r'\n+', ' ', text)
    # Remove resulting ' '
    text = text.strip()
    text = re.sub(r'\s\s+', ' ', text)

    # emails
    text = re.sub('\S*@\S*\s?', '', text)

    # > Quotes
    text = re.sub(r'\"?\\?&?gt;?', '', text)

    # Bullet points/asterisk (bold/italic)
    text = re.sub(r'\*', '', text)
    text = re.sub('&amp;#x200B;', '', text)

    # things in parantheses or brackets
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)

    # remove URLS
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)

    # Strikethrough
    text = re.sub('~', '', text)

    # Spoiler, which is used with < less-than (Preserves the text)
    text = re.sub('&lt;', '', text)
    text = re.sub(r'!(.*?)!', r'\1', text)

    # Code, inline and block
    text = re.sub('`', '', text)

    # Superscript (Preserves the text)
    text = re.sub(r'\^\((.*?)\)', r'\1', text)

    # Table
    text = re.sub(r'\|', ' ', text)
    text = re.sub(':-', '', text)

    # Heading
    text = re.sub('#', '', text)

    return text
