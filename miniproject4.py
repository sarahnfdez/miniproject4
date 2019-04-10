# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:47:17 2019

@author: Sarah
"""
import csv
import numpy as np
import textdistance
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords

class corpus_classifier:
    def __init__(self):
        return None
    
    def load_training_data(self):
        X_training = []
        y_training = []
        
        with open("snli_1.0_train.txt") as f:
            reader = csv.reader(f, delimiter="\t")
            d = list(reader)
        df = d[1:]
        
        #loads x training
        for arr in df:
            sentences = arr[5:7]
            X_training.append(sentences)
            
        #Loads y training
        for arr in df:
            element = arr[0]
            if(element != 'entailment'):
                y_training.append(0)
            else:
                y_training.append(1)
        
        return X_training, y_training
    
    def load_validation_data(self):
        X_valid = []
        y_valid = []
        
        with open("snli_1.0_dev.txt") as f:
            reader = csv.reader(f, delimiter="\t")
            d = list(reader)
        df = d[1:]
        
        #loads x training
        for arr in df:
            sentences = arr[5:7]
            X_valid.append(sentences)
            
        #Loads y training
        for arr in df:
            element = arr[0]
            if(element != 'entailment'):
                y_valid.append(0)
            else:
                y_valid.append(1)
        
        return X_valid, y_valid
    
    def load_test_data(self):
        X_test = []
        y_test = []
        
        with open("snli_1.0_test.txt") as f:
            reader = csv.reader(f, delimiter="\t")
            d = list(reader)
        df = d[1:]
        
        #loads x training
        for arr in df:
            sentences = arr[5:7]
            X_test.append(sentences)
            
        #Loads y training
        for arr in df:
            element = arr[0]
            if(element != 'entailment'):
                y_test.append(0)
            else:
                y_test.append(1)
        
        return X_test, y_test
    
    def edit_distance_preprocess(self, x):
        stop_words = set(stopwords.words('english'))
        for pair in x:
            for sentence in pair:
                sentence = sentence.lower()
                sentence = word_tokenize(sentence)
                sentence = [w for w in sentence if not w in stop_words]
                
        for pair in x:
            for sentence in pair:
                final = ''
                for word in sentence:
                    final += word
                sentence = final
        
        return x
    
    def jaro_winkler_train(self, x, y):
        x_train = []
        for pair in x:
            dist = textdistance.jaro_winkler(pair[0], pair[1])
            x_train.append(dist)
            
        return x
    
    def levenshtein_train(self, x, y):
        x_train = []
        for pair in x:
            dist = textdistance.levenshtein.normalized_similarity(pair[0], pair[1])
            x_train.append(dist)
        
        return x
            
    def hamming_train(self, x, y):
        x_train = []
        for pair in x:
            dist = textdistance.hamming.normalized_similarity(pair[0], pair[1])
            x_train.append(dist)
            
        return x
        
        
        
        
        
        
        
    
#############################################################