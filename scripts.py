#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:14:46 2018

@author: tron
"""

import pickle

pkl = open('vectorizer.pickle', 'rb')
tf_vect = pickle.load(pkl)

vec = open('mlmodel.pickle', 'rb')
clf = pickle.load(vec)



#Random state 42
#review_train, review_test, label_train, label_test = train_test_split(result['pos'], result['Labels'], test_size=0.2, random_state=42)

#X_test_tf = tf_vect.transform(review_test)
#pred = clf.predict(X_test_tf)

#print(metrics.accuracy_score(label_test, pred))

#print(confusion_matrix(label_test, pred))

#print(classification_report(label_test, pred))

def test_string(s):
    X_test_tf = tf_vect.transform([s])
    y_predict = clf.predict(X_test_tf)
    return y_predict

print(test_string("The hotel was good.The room had a 27-inch Samsung led tv, a microwave.The room had a double bed")
)

print(test_string("My family and I are huge fans of this place. The staff is super nice, and the food is great. The chicken is very good, and the garlic sauce is perfect. Ice cream topped with fruit is delicious too. Highly recommended!")
)