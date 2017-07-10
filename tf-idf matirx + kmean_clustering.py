# coding: utf-8
#!/usr/bin/env python

# I ran the program in the juypter. There might be some errors if you try to
# run it in the IDEL. 

#------------------------------------------------------------------------------#
#             Create CSV file of DF-IDF values for all diagnoses               # 
#------------------------------------------------------------------------------#

import codecs
import os
import sys
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt
import csv

def get_document_filenames(document_path="symptom"):
    # change the path of the document to where you save the symptoms txt folder
    # For windows, you can just say symptom (document_path="symptom")
    # if this code is saved at the same place with symptom folder
    return [os.path.join(document_path, each) for each in os.listdir(document_path)]

def display_scores(vectorizer, tfidf_result):
    # this method is used to display the features from most common to least common
    # based on tf-idf values
    scores = zip(vectorizer.get_feature_names(),np.asarray(tfidf_result.mean(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores:
        print("{0:50} Score: {1}".format(item[0], item[1]))

#------------------------------------------------------------------------------#
#  Define method to return the elbow plot to determine the optimal# of cluster # 
#------------------------------------------------------------------------------#
def elbow(df,n):
    # the input is a numpy dataframe and n is the max number of cluster we want
    # Let's say 40
    # generate the elbow plot
    # X axis is number of clusters
    # Y axis is the percent of variance explained by each number of cluster
    # Objective is finding out the elbow point
    kRanges = range(1,n)
    kMeansVar = [KMeans(n_clusters=k).fit(df.values) for k in kRanges]
    centroids = [X.cluster_centers_ for X in kMeansVar]
    k_euclid = [cdist(df.values, cent) for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(df.values)**2)/df.values.shape[0]
    bss = tss - wcss
    fig = plt.figure()
    plt.plot(kRanges, bss/tss*100)
    plt.title('Elbow for KMeans clustering')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained (%)')
    plt.show()

def elbow_awss(df, n):
    # the input is a numpy dataframe and n is max number of cluster we want 
    # generate the elbow plot
    # X axis is still number of clusters
    # Y axis is the average sum of square distance within clusters
    # Objective is finding out the elbow point
    K = range(1,n)
    KM = [KMeans(n_clusters=k).fit(df) for k in K]
    centroids = [k.cluster_centers_ for k in KM]
    D_k = [cdist(df, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D,axis=1) for D in D_k]
    dist = [np.min(D,axis=1) for D in D_k]
    avgWithinSS = [sum(d)/df.shape[0] for d in dist]
    # Total with-in sum of square
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(df)**2)/df.shape[0]
    bss = tss-wcss
    
    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, avgWithinSS, 'b*-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')
    plt.show()


#------------------------------------------------------------------------------#
#                          Starting the main from here                         # 
#------------------------------------------------------------------------------#

def main():
    tfidf_vectorizer = TfidfVectorizer(input='filename',
                                       max_features=500,
                                       stop_words='english',
                                       ngram_range=(3, 6),
                                       #change the n-gram size here 
                                       decode_error = 'ignore')

    tf_idf_matrix = tfidf_vectorizer.fit_transform(get_document_filenames())
    #finds the tfidf score with normalization
    df = pd.DataFrame(tf_idf_matrix.todense())
    
    #You can either export the matrix as the csv file
    #Code: df.to_csv("tf-idf_more3_.csv") 
    
    #Or you can print out most common word chunks and their tfidf values
    #Code: display_scores(tfidf_vectorizer, tf_idf_matrix)
    
    #Or just print out the matrix as a nicer format 
    #Code: print(tf_idf_matrix.todense())


#------------------------------------------------------------------------------#
#           Convert each row to vector(array) for distance calculation         # 
#------------------------------------------------------------------------------#

    #convert each row of the csv file into a (Euclidian)vector(array) for distance calculation
    array_lst = []
    with open("tf-idf_all_pure#.csv","r") as csvfile:
        # delete the header and the first column of the code
        # only maintain the matrix of values
        # but the paste the first column and header(first row) in another file
        # DO NOT DELETE THEM!
        # We will paste them back after the program finished running 
        reader = csv.reader(csvfile)
        for row in reader:
            my_lst = row
            array_lst.append(my_lst)
            #generate a list whose elements are lists of row's values
    X = np.array(array_lst)
    # converting the array_list as vectors(array)
    df = pd.DataFrame(X)
    # save the array as a data frame 
    elbow_awss(df,40)
    elbow(df,40)
    
    # display the elbow plot of the data frame

    kmeans = KMeans(n_clusters=40).fit(X)
    # After figuring out the number of cluster, do K-mean clustering
    
    df_labels = pd.DataFrame(kmeans.labels_)
    # df_labels.to_csv("cluster.csv")
    # generate the label of the cluster and paste the label as first column
    # BEFORE SORTING!! paste back the header(first row of word chunks)
    # Paste back the column of codes as the second column 

    # Order of the column:
    # 1. cluster label
    # 2. code
    # 3-***. each column has specific word chunks all tfidf value

    # Then sort based on the cluster label, excluding the header(first row)
    # DONE!

main()

#------------------------------------------------------------------------------#
#                                End   of  Story                               # 
#------------------------------------------------------------------------------#
