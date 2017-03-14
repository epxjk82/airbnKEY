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

def get_top_term_frequency(tf, df, column, by, lower_lim, upper_lim, num_words=25):
    segment_df = df[(df[by]>=lower_lim) & (df[by]<upper_lim)]
    prop_cnt = segment_df.shape[0]
    print "Number of properties: {}".format(prop_cnt)

    tf_matrix = tf.fit_transform(segment_df[column])
    tf_vocab= np.array(tf.get_feature_names())
    tf_matrix_sum = np.sum(tf_matrix.toarray(),axis=0)
    sorted_ind = np.argsort(tf_matrix_sum)[::-1]

    print "Top term frequency based on {}, {} > {} > {}".format(by,lower_lim, by, upper_lim)
    print "---------------------------------------------"
    for word, count in zip(tf_vocab[sorted_ind[:num_words]], tf_matrix_sum[sorted_ind[:num_words]]):
        print "{:>30}, {:>4d}, {:.2f}%".format(word, count, (count*1.0/prop_cnt)*100)



def get_top_tfidf_words(df, column, tfidf_matrix, tfidf_vocab, start_prop=0, end_prop=100, num_words=25, bot=False):

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
