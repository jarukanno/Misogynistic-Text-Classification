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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from newmm_tokenizer.tokenizer import word_tokenize
from attacut import tokenize
from sklearn import svm
from sklearn import tree
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

#svm
svm_clf = svm.SVC()
svm_clf.fit(x_train_tf,Ytrain_list)
svm_pred = svm_clf.predict(x_test_t)



# #build confusion matrix
svm_matrix = confusion_matrix(y_true=y_test_list, y_pred=svm_pred)


#logistic Regrssion
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train_tf, Ytrain_list)
logisticRegr_pred = logisticRegr.predict(x_test_t)

#build confusion matrix
logisticRegr_matrix = confusion_matrix(y_true=y_test_list, y_pred=logisticRegr_pred)


# DecisionTree
dec = tree.DecisionTreeClassifier() # defining decision tree classifier
dec.fit(x_train_tf, Ytrain_list) # train data on new data and new target
dec_pred = dec.predict(x_test_t) #  assign removed data as input

#build confusion matrix
dec_matrix = confusion_matrix(y_true=y_test_list, y_pred=dec_pred )

def show_matrix(model):
    plt.clf()
    plt.imshow(model, interpolation='nearest', cmap=plt.cm.Pastel2)
    classNames = ['Non-Misogynist','Misogynist']
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=0)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(model[i][j]))
    plt.show()

def show_score(model_pred):
    print('Accuracy Score: %.3f' % accuracy_score(y_test, model_pred))
    print('F1 Score: %.3f' % f1_score(y_test, model_pred))
    print('Recall: %.3f' % recall_score(y_test, model_pred))
    print('Precision: %.3f' % precision_score(y_test, model_pred))

# show_score(dec_pred)
# show_matrix(dec_matrix)


joblib.dump(svm_clf, 'model/SVM_text_clf.pkl')
joblib.dump(logisticRegr, 'model/logisticRegr_text_clf.pkl')
joblib.dump(dec, 'model/decision_text_clf.pkl')




def getCountVectorModel():
    return count_v

def getTFtranformer():
    return tf_transformer