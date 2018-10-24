#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 21:22:31 2018

@author: tron
"""

import os     #for iterating over the folders
import fnmatch   #filtering only the text files
from textblob import TextBlob
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag, pos_tag_sents
import re
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import pickle

path = 'op_spam_train/'

label = []

configfiles = [os.path.join(subdir, f) 
for subdir, dirs, files in os.walk(path)
    for f in fnmatch.filter(files, '*.txt')]

print(len(configfiles))

for f in configfiles:
    c = re.search('(trut|deceptiv)\w', f)
    label.append(c.group())
    
print(label)

labels = pd.DataFrame(label, columns=['Labels'])

print(labels.head(5))

review = []
directory = os.path.join('op_spam_train/')
for subdir, dirs, files in os.walk(directory):
    for file in files:
        if fnmatch.filter(files, '*.txt'):
            f = open(os.path.join(subdir, file),'r')
            a = f.read()
            review.append(a)
            
reviews = pd.DataFrame(review, columns=['HotelReviews'])
print(reviews.head())

result = pd.merge(reviews, labels, right_index=True, left_index=True)
result['HotelReviews'] = result['HotelReviews'].map(lambda x: x.lower())

print(result.head())

stop = stopwords.words('english')

result['review_without_stopwords'] = result['HotelReviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

print(result.head())

def pos(review_without_stopwords):
    return TextBlob(review_without_stopwords).tags

os = result.review_without_stopwords.apply(pos)
os1 = pd.DataFrame(os)

print(os1.head())

os1['pos'] = os1['review_without_stopwords'].map(lambda x: " ".join(["/".join(x) for x in x ]))

result = pd.merge(result, os1, right_index=True, left_index=True)

print(result.head())

review_train, review_test, label_train, label_test = train_test_split(result['pos'], result['Labels'], test_size=0.2, random_state=13)

tf_vect = TfidfVectorizer(lowercase = True, use_idf=True, smooth_idf=True, sublinear_tf=False)

X_train_tf = tf_vect.fit_transform(review_train)
X_test_tf = tf_vect.transform(review_test)

def svm_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_

svm_param_selection(X_train_tf, label_train, 5)

clf = svm.SVC(C=10, gamma=0.001, kernel='linear')
clf.fit(X_train_tf, label_train)
pred = clf.predict(X_test_tf)

with open('vectorizer.pickle', 'wb') as fin:
    pickle.dump(tf_vect, fin)
    
with open('mlmodel.pickle', 'wb') as f:
    pickle.dump(clf, f)
    




    





            
            

