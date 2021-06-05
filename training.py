import numpy as np
import pandas as pd
import matplotlib
import sklearn
import sys
import os
import json
import io
import warnings
warnings.filterwarnings("ignore")
import re
import string
import pythainlp
import matplotlib.pyplot as plt
import seaborn as sns
import data_pre_processing as prep_data

from pythainlp.util import *

from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.corpus import words
from stop_words import get_stop_words
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from newmm_tokenizer.tokenizer import word_tokenize
from attacut import tokenize
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
import joblib


stop = thai_stopwords()

X_train,X_test,y_train,y_test = train_test_split(prep_data.dataset[['tweet_text']],prep_data.dataset[['Misogyny']],test_size=0.3,random_state=7)

Xtrain_list = []
for index, row in X_train.iterrows():
    
    Xtrain_list.append(row["tweet_text"])

Ytrain_list = []
for index, row in y_train.iterrows():
    Ytrain_list.append(row["Misogyny"])

x_test_list = []
for index, row in X_test.iterrows():
    x_test_list.append(row["tweet_text"])

y_test_list = []
for index, row in y_test.iterrows():
    y_test_list.append(row["Misogyny"])


count_v = CountVectorizer(stop_words=stop,tokenizer=word_tokenize)
count_train = count_v.fit_transform(Xtrain_list)

tf_transformer = TfidfTransformer(norm='l2')
tf_transformer.fit(count_train)
x_train_tf = tf_transformer.transform(count_train)

x_test_c = count_v.transform(x_test_list)
x_test_t = tf_transformer.transform(x_test_c)

#naive Bayes
nb_clf = MultinomialNB()
nb_clf.fit(x_train_tf, Ytrain_list)
nb_pred = nb_clf.predict(x_test_t) # store the prediction data
nb_ac = accuracy_score(y_test_list,nb_pred) # calculate the accuracy


#build confusion matrix
# nb_matrix = confusion_matrix(y_true=y_test_list, y_pred=nb_pred)
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.matshow(nb_matrix, cmap=plt.cm.Oranges, alpha=0.3)
# for i in range(nb_matrix.shape[0]):
#     for j in range(nb_matrix.shape[1]):
#         ax.text(x=j, y=i,s=nb_matrix[i, j], va='center', ha='center', size='xx-large')
 
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix of NB', fontsize=18)
# plt.show()

#svm
svm_clf = svm.SVC()
svm_clf.fit(x_train_tf,Ytrain_list)
svm_pred = svm_clf.predict(x_test_t)
svm_ac = accuracy_score(y_test_list,svm_pred)


# #build confusion matrix
# svm_matrix = confusion_matrix(y_true=y_test_list, y_pred=svm_pred)
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.matshow(svm_matrix, cmap=plt.cm.Greens, alpha=0.3)
# for i in range(svm_matrix.shape[0]):
#     for j in range(svm_matrix.shape[1]):
#         ax.text(x=j, y=i,s=svm_matrix[i, j], va='center', ha='center', size='xx-large')
 
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix of SVM', fontsize=18)
# plt.show()
# print("accuray score of SVM model is: ",svm_ac)

#logistic Regrssion
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train_tf, Ytrain_list)
logisticRegr_pred = logisticRegr.predict(x_test_t)

logisticRegr_ac = accuracy_score(y_test_list,logisticRegr_pred)

#build confusion matrix
# logisticRegr_matrix = confusion_matrix(y_true=y_test_list, y_pred=logisticRegr_pred)
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.matshow(logisticRegr_matrix, cmap=plt.cm.Blues, alpha=0.3)
# for i in range(logisticRegr_matrix.shape[0]):
#     for j in range(logisticRegr_matrix.shape[1]):
#         ax.text(x=j, y=i,s=logisticRegr_matrix[i, j], va='center', ha='center', size='xx-large')
 
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix of Logistic Regression', fontsize=18)
# plt.show()
# print("accuray score of Logistic regression is : ",logisticRegr_ac)


joblib.dump(nb_clf, 'model/NB_text_clf.pkl')
joblib.dump(svm_clf, 'model/SVM_text_clf.pkl')
joblib.dump(logisticRegr, 'model/logisticRegr_text_clf.pkl')


def getNBScore():
    return nb_ac

def getSVMScore():
    return svm_ac

def getlogisticRegrScore():
    return logisticRegr_ac

def getCountVectorModel():
    return count_v

def getTFtranformer():
    return tf_transformer