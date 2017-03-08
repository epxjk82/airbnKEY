import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt

airbnb_stopwords = [
'and', 'the', 'to', 'in', 'of', 'with', 'is',  'on', 'you', 'this','our', 'has', 'are', 'for','your', 'out', 'there', 'will',
'can', 'be',  'but', 'its', 're','which','here', 'or',  'we', 'it',  'an','from',
'room','bedroom','bed',
]

def get_top_term_frequency(df, by, column, lower_lim, upper_lim, ngram_range=(2,2), num_words=25, stop_words=airbnb_stopwords):
    segment_df = df[(df[column]>=lower_lim) & (df[column]<upper_lim)]
    prop_cnt = segment_df.shape[0]
    print "Number of properties: {}".format(prop_cnt)

    tf = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range, max_features=5000)
    tf_matrix = tf.fit_transform(segment_df[by])
    tf_vocab= np.array(tf.get_feature_names())
    tf_matrix_sum = np.sum(tf_matrix.toarray(),axis=0)
    sorted_ind = np.argsort(tf_matrix_sum)[::-1]

    print "Top term frequency based on {}, {} > {} > {}".format(by,lower_lim,column, upper_lim)
    print "---------------------------------------------"
    for word, count in zip(tf_vocab[sorted_ind[:num_words]], tf_matrix_sum[sorted_ind[:num_words]]):
        print "{:>30}, {:>4d}, {:.2f}%".format(word, count, (count*1.0/prop_cnt)*100)



def get_top_tfidf_words(df, column, tfidf_matrix, tfidf_vocab, num_prop=50, num_words=25, bot=False):
    if bot:
        num_prop_ind = np.argsort(df[column])[::-1][-num_prop:]
    else:
        num_prop_ind = np.argsort(df[column])[::-1][:num_prop]

    num_prop_tfidf_sum = np.sum(tfidf_matrix.toarray()[num_prop_ind], axis=0)
    num_prop_tfidf_cnt = np.count_nonzero(tfidf_matrix.toarray()[num_prop_ind], axis=0)
    num_prop_tfidf_mean = num_prop_tfidf_sum/num_prop_tfidf_cnt
    num_prop_tfidf_mean_sorted_ind = np.argsort(num_prop_tfidf_mean)[::-1][:num_words]
    if bot:
        order_type = 'bot'
    else:
        order_type = 'top'

    num_prop_tfidf_top_vocab = tfidf_vocab[num_prop_tfidf_mean_sorted_ind]
    num_prop_tfidf_top_vocab_score = num_prop_tfidf_mean[num_prop_tfidf_mean_sorted_ind]
    print "Top {} tfidf words for {} {} properties by {}".format(num_words, order_type,num_prop, column)
    print "----------------------------------------------------"
    for word, tfidf_mean in zip(num_prop_tfidf_top_vocab,num_prop_tfidf_top_vocab_score):
        print "{:>20}: {:>.3f}".format(word, tfidf_mean)
