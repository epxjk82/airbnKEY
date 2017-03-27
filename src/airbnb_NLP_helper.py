import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

import re
import matplotlib.pyplot as plt

sb_stemmer = SnowballStemmer("english")
p_stemmer = PorterStemmer()

airbnb_stopwords = [
'and', 'the', 'to', 'in', 'of', 'with', 'is',  'on', 'you', 'this','our', 'has', 'are', 'for','your', 'out', 'there', 'will',
'can', 'be',  'but', 'its', 're','which','here', 'or',  'we', 'it',  'an','from',
'room','bedroom','bed',
]

def tokenize_and_stem(text):
    """Tokenize and stem words

    Parameters
    ----------
    text : corpus (array of documnetrs)

    Returns
    -------
    stems : list of stemmed words
    """
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [sb_stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    """Tokenize words to alphabetical only

    Parameters
    ----------
    text : corpus (array of documents)

    Returns
    -------
    stems : list of filtered words
    """

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


def get_top_term_frequency(tf, df, column, by, lower_lim, upper_lim, num_words=25):
    """ Retrieve and print top words by term frequency

    Parameters
    ----------
    tf : CounVectorizer term frequency array
    df : pandas df for corpus
    column: str
        column in df containing corpus
    by: str
        column to filter by
    lower_lim: int
        lower limit of df[by]
    upper_lim: int
        upper limit of df[by]
    num_words: int

    Returns
    -------
    None
    """

    segment_df = df[(df[by]>=lower_lim) & (df[by]<upper_lim)]
    prop_cnt = segment_df.shape[0]
    print "Number of properties: {}".format(prop_cnt)

    tf_matrix = tf.fit_transform(segment_df[column])
    tf_vocab= np.array(tf.get_feature_names())
    tf_matrix_sum = np.sum(tf_matrix.toarray(),axis=0)
    sorted_ind = np.argsort(tf_matrix_sum)[::-1]

    print "Top term frequency based on {}, {} > {} > {}".format(by,lower_lim, by, upper_lim)
    print ""
    print "{:>30}, {:>4}, {:>5}".format("WORD", "COUNT", "PCT")
    print "---------------------------------------------"
    for word, count in zip(tf_vocab[sorted_ind[:num_words]], tf_matrix_sum[sorted_ind[:num_words]]):
        print "{:>30}, {:>5d}, {:>.2f}%".format(word, count, (count*1.0/prop_cnt)*100)



def get_top_tfidf_words(df, column, tfidf_matrix, tfidf_vocab, start_prop=0, end_prop=100, num_words=25):
    """ Print out top words for a given range of properties based on tfidf values

    Parameters
    ----------
    df : pandas DataFrame
        full dataset
    column: str
        column name containing corpus
    tfidf_matrix:
        sparse matrix object from TfidfVectorizer
    tfidf_vocab:
        numpy array of vocabulary from TfidfVectorizer
    start_prop: int
        starting property index
    end_prop: int
        ending property index
    num_words: int
        number of words to display


    Returns
    -------
    None
    """
    ind = np.argsort(df[column])[::-1][start_prop:end_prop]

    num_prop_tfidf_sum = np.sum(tfidf_matrix.toarray()[ind], axis=0)
    num_prop_tfidf_cnt = np.count_nonzero(tfidf_matrix.toarray()[ind], axis=0)
    num_prop_tfidf_cnt[num_prop_tfidf_cnt==0] = 1.
    num_prop_tfidf_mean = num_prop_tfidf_sum/num_prop_tfidf_cnt
    #num_prop_tfidf_mean = np.mean(tfidf_matrix.toarray()[ind], axis=0)
    num_prop_tfidf_mean_sorted_ind = np.argsort(num_prop_tfidf_mean)[::-1][:num_words]

    num_prop_tfidf_top_vocab = tfidf_vocab[num_prop_tfidf_mean_sorted_ind]
    num_prop_tfidf_top_vocab_score = num_prop_tfidf_mean[num_prop_tfidf_mean_sorted_ind]
    print "Top {} tfidf words for properties in range {}-{} by {}".format(num_words, start_prop, end_prop, column)
    print "----------------------------------------------------"
    for word, tfidf_mean in zip(num_prop_tfidf_top_vocab,num_prop_tfidf_top_vocab_score):
        print "{:>30}: {:>.3f}".format(word, tfidf_mean)

def get_top_N_words_per_kmeans_cluster(cluster_centers, vocab, n_words=10):
    """ Print out top words per KMeans cluster

    Parameters
    ----------
    cluster_centers : numpy array
        cluster centers from a KMeans model
    vocab: numpy array
        vocabulary of words
    n_words: int
        number of top words to print

    Returns
    -------
    topN_words_list : list
        list of top n_words
    """
    topN_tfidf_list = []
    topN_words_list = []
    for cluster in cluster_centers:
        sorted_ind = np.argsort(cluster)[::-1][:n_words]
        topN_words_list.append(vocab[sorted_ind])
        topN_tfidf_list.append(cluster[sorted_ind])

    return topN_words_list

def get_topic_weights_prop_range(df, by, W, H, lower_lim, upper_lim, num_words=5):
    """ Print out top words per topic for NMF model

    Parameters
    ----------
    df : pandas DataFrame
        full dataset
    by: str
        column to filter by
    W: pandas DataFrame
        topic weights matrix from NMF
    H: pandas DataFrame
        topic-to-words mapping matrix from NMF
    lower_lim: int
        lower bound of df[column]
    upper_lim: int
        upper bound of df[column]
    num_words: int
        number of words to print per topic

    Returns
    -------
    None
    """
    prop_ind = list(df.reset_index()[(df.reset_index()[by]>lower_lim) & (df.reset_index()[by]<upper_lim)].index)
    print "n = ", len(prop_ind)
    topic_weights = np.mean(W.iloc[prop_ind], axis=0)
    sorted_ind = np.argsort(topic_weights, axis=0)[::-1]

    print '{:>8}, {:>5}, {:>5}, {}'.format('TOPIC', 'WT', 'SUMWT', 'WORDS')
    weight_cumsum=0
    for ind in sorted_ind:
        weight_cumsum +=topic_weights[ind]
        top_words = ', '.join(list(H.iloc[ind].sort_values(ascending=False).index[:num_words])).encode('utf-8')
        print '{:>8}, {:.03f}, {:.03f}, {}'.format(ind, topic_weights[ind],weight_cumsum,top_words)
