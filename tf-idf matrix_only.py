# coding: utf-8
#!/usr/bin/env python
import codecs
import os
import sys
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def get_document_filenames(document_path="symptom"):
    # return the list of path of the file 
    return [os.path.join(document_path, each) for each in os.listdir(document_path)]

def display_scores(vectorizer, tfidf_result):
    
    scores = zip(vectorizer.get_feature_names(),np.asarray(tfidf_result.mean(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    #for item in sorted_scores:
     #   print("{0:50} Score: {1}".format(item[0], item[1]))
        

def main():
    tfidf_vectorizer = TfidfVectorizer(input='filename',
                                       max_features=500,
                                       stop_words='english',
                                       ngram_range=(1, 6),
                                       decode_error = 'ignore')
    

    tf_idf_matrix = tfidf_vectorizer.fit_transform(get_document_filenames())
    #finds the tfidf score with normalization

    #display_scores(tfidf_vectorizer, tf_idf_matrix)
    
    matrix = tf_idf_matrix.todense()
    
    df = pd.DataFrame(matrix)
    df.to_csv("tf-idf_all.csv")
    
    

main()

# Prepend vector from get_document_filenames() onto tf_idf_matrix result
# Prepend row from vectorizer.get_feature_names() (with one blank cell at the beginning) onto ^^


