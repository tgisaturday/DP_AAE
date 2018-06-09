import re
import time
import logging
import numpy as np
import pandas as pd
import random
import json
import sys
import os
from nltk.corpus import stopwords
from collections import Counter
from contractions import get_contractions
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords

def empty_remover(text):
    removed = []
    for word in text:
        if word != '':
            removed.append(word)
    return removed

def clean_str(text,max_length):
    """Clean sentence"""
    text = text.lower()
    text = text.split()
    new_text = []
    contractions = get_contractions()
    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    text = " ".join(new_text)
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'[0-9]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    text = text.split(' ')
    #stops = set(stopwords.words("english"))
    #text = [w for w in text if not w in stops]    
    text = empty_remover(text)

    if len(text) >= max_length:
        text = text[0:max_length]
    elif len(text) < max_length:
        text = text + ["PAD"] * (max_length - len(text))
        text = text[0:max_length]
        
    return ' '.join(text).strip()

def gen_target(text,max_length):
    """Clean sentence"""
    text = text.lower()
    text = text.split()
    new_text = []
    contractions = get_contractions()
    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    text = " ".join(new_text)
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'[0-9]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = text.split(' ')
    #stops = set(stopwords.words("english"))
    #text = [w for w in text if not w in stops]
    text = ['GO']+text
    text = empty_remover(text)
    if len(text) >= max_length:
        text = text[0:max_length]
    else:
        text = text + ["PAD"] * (max_length - len(text))
        text = text[0:max_length]
    return ' '.join(text)
    
def load_data(filename,dataset_name,max_length):
    """Load sentences and labels"""
    if dataset_name == 'ag_news' or dataset_name == 'dbpedia' or dataset_name == 'sogou_news' or dataset_name == 'amazon_review_full' or dataset_name == 'amazon_review_polarity' :
        df = pd.read_csv(filename, names=['label', 'title', 'text'], dtype={'title': object,'text': object})
        selected = ['text']
        non_selected = list(set(df.columns) - set(selected))
        df = df.drop(non_selected, axis=1) # Drop non selected columns 
        df = df.reindex(np.random.permutation(df.index)) # Shuffle the dataframe
        x_raw = df['text'].apply(lambda x: clean_str(x,max_length-1)).tolist()
        target_raw = df['text'].apply(lambda x: gen_target(x,max_length)).tolist()
        
    elif dataset_name == 'yelp_review_full' or dataset_name == 'yelp_review_polarity':
        df = pd.read_csv(filename, names=['label','text'], dtype={'text': object})
        selected = ['text']
        non_selected = list(set(df.columns) - set(selected))
        df = df.drop(non_selected, axis=1) # Drop non selected columns        
        x_raw = df['text'].apply(lambda x: clean_str(x,max_length-1)).tolist()
        target_raw = df['text'].apply(lambda x: gen_target(x,max_length)).tolist()         
            
    elif dataset_name == 'yahoo_answers':
        df = pd.read_csv(filename, names=['label', 'title', 'content','answer'], dtype={'title': object,'answer': object,'content': object})
        selected = ['content','answer']     
        non_selected = list(set(df.columns) - set(selected))
        df = df.drop(non_selected, axis=1) # Drop non selected columns        
        df['temp'] = df[['content','answer']].apply(lambda x: ' '.join(str(v) for v in x), axis=1)
        x_raw = df['temp'].apply(lambda x: clean_str(x,max_length-1)).tolist()
        target_raw = df['temp'].apply(lambda x: gen_target(x,max_length)).tolist()         
    return x_raw,target_raw

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Iterate the data batch by batch"""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
            
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            
            yield shuffled_data[start_index:end_index]


