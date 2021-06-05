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
import preprocessor as p
import pythainlp

from pythainlp.util import *

from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.corpus import words
from stop_words import get_stop_words


df = pd.read_csv('dataset/misogynyst_dataset.csv')
df.drop('tweet_id', inplace=True, axis=1)
dataset = df.dropna()
dataset = dataset.drop(dataset[dataset.tweet_text == "TWEET_NOT_FOUND"].index)
dataset = dataset.astype({"Misogyny":'int'}) 

def clean_msg(msg):
    
    #  ลบ url
    p.set_options(p.OPT.URL)
    msg = p.clean(''+msg+'') 
    
    # ลบ separator เช่น \n \t
    msg = ' '.join(msg.split())
    
    msg = re.sub(r'^RT[\s]+', '', msg)
   
    # ลบ hashtag
    msg = re.sub(r'#','',msg)
    
    msg = re.sub(r'“','',msg)
    
    msg = re.sub(r'”','',msg)
    
    msg = re.sub(r'—','',msg)

    msg = re.sub(r'[a-zA-Z]', '', msg)
    
    # ลบ เครื่องหมายคำพูด (punctuation)
    for c in string.punctuation:
        msg = re.sub(r'\{}'.format(c),'',msg)
    
    msg = msg.replace(" ", "") 
    
    msg = msg.replace("\n", "")
    
    #ลบตัวเลข
    
    msg = ''.join(filter(lambda x: not x.isdigit(), msg))
    
    #ลบ emoji
    
    
    msg = remove_emoji(msg)
    
  
    
    
    return msg


def remove_emoji(data):

    if not data:
        return data
    if not isinstance(data, str):
        return data
    try:
       patt = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    except re.error:
       patt = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
    return patt.sub('', data)


for index, row in dataset.iterrows():
    dataset.at[index,'tweet_text']= clean_msg(row["tweet_text"])


