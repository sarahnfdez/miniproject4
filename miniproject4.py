# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:47:17 2019

@author: Sarah
"""
import csv
import textdistance
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords

class corpus_classifier:
    def __init__(self):
        return None
    
    def load_training_data(self):
        X_training = [550152]
        x_index = 0
        y_training = [550152]
        y_index = 0
        
        with open("snli_1.0_train.txt") as f:
            reader = csv.reader(f, delimiter="\t")
            d = list(reader)
        df = d[1:]
        
        #loads x training
        for arr in df:
            sentences = arr[5:7]
            X_training[x_index] = sentences
            x_index += 1
            
        #Loads y training
        for arr in df:
            element = arr[0]
            if(element != 'entailment'):
                y_training[y_index] = 0
                y_index += 1
            else:
                y_training[y_index] = 1
                y_index += 1
        
        return X_training, y_training
    
    def load_validation_data(self):
        X_val = [550152]
        x_index = 0
        y_val = [550152]
        y_index = 0
        
        with open("snli_1.0_dev.txt") as f:
            reader = csv.reader(f, delimiter="\t")
            d = list(reader)
        df = d[1:]
        
        #loads x training
        for arr in df:
            sentences = arr[5:7]
            X_val[x_index] = sentences
            x_index += 1
            
        #Loads y training
        for arr in df:
            element = arr[0]
            if(element != 'entailment'):
                y_val[y_index] = 0
                y_index += 1
            else:
                y_val[y_index] = 1
                y_index += 1
        
        return X_val, y_val
    
    def load_test_data(self):
        X_test = [550152]
        x_index = 0
        y_test = [550152]
        y_index = 0
        
        with open("snli_1.0_test.txt") as f:
            reader = csv.reader(f, delimiter="\t")
            d = list(reader)
        df = d[1:]
        
        #loads x training
        for arr in df:
            sentences = arr[5:7]
            X_test[x_index] = sentences
            x_index += 1
            
        #Loads y training
        for arr in df:
            element = arr[0]
            if(element != 'entailment'):
                y_test[y_index] = 0
                y_index += 1
            else:
                y_test[y_index] = 1
                y_index += 1
        
        return X_test, y_test
    
    def edit_distance_preprocess(self, x):
        stop_words = set(stopwords.words('english'))
        preprocessed1 = [550152]
        pi = 0
        for pair in x:
            finalpair = [2]
            fi = 0
            for sentence in pair:
                sentence = sentence.lower()
                sentence = word_tokenize(sentence)
                sentence = [w for w in sentence if not w in stop_words]
                finalpair[fi] = sentence
                fi += 1
            preprocessed1[pi] = finalpair
            pi += 1
            
        preprocessed2 = [550152]
        pi = 0
        for pair in preprocessed1:
            finalpair= [2]
            fi = 0
            for sentence in pair:
                final = ''
                for word in sentence:
                    if(word != '.'):
                        final += word
                        final += ' '
                    if(word == '.'):
                        final = final[:-1]
                finalpair[fi] = final
                fi += 1
            preprocessed2[finalpair]
            pi += 1
        
        return preprocessed2
    
    def jaro_winkler_classifier(self, x_train, y_train, x_test):
        x_dist = [550152]
        y_test = [10000]
        
        xd_index = 0
        for pair in x_train:
            dist = textdistance.jaro_winkler(pair[0], pair[1])
            x_dist[xd_index] = dist
            xd_index += 1
        
        yt_index= 0
        for pair in x_test:
            dist = textdistance.jaro_winkler(pair[0], pair[1])
            diff = 10000
            index = 0
            for train_dist in x_dist:
                if(abs(train_dist-dist) < diff):
                    diff = abs(train_dist-dist)
                    index = x_dist.index(train_dist)
            y_test[yt_index] = y_train[index]
            yt_index += 1
        
        y_test = y_test[1:]
        return y_test
    
    def levenshtein_classifier(self, x_train, y_train, x_test):
        x_dist = [550152]
        y_test = [10000]
        
        xd_index = 0
        for pair in x_train:
            dist = textdistance.levenshtein.normalized_similarity(pair[0], pair[1])
            x_dist[xd_index] = dist
            xd_index += 1
        
        yt_index= 0
        for pair in x_test:
            dist = textdistance.jaro_winkler(pair[0], pair[1])
            diff = 10000
            index = 0
            for train_dist in x_dist:
                if(abs(train_dist-dist) < diff):
                    diff = abs(train_dist-dist)
                    index = x_dist.index(train_dist)
            y_test[yt_index] = y_train[index]
            yt_index += 1
        
        y_test = y_test[1:]
        return y_test
    
    def hamming_classifier(self, x_train, y_train, x_test):
        x_dist = [550152]
        y_test = [10000]
        
        xd_index = 0
        for pair in x_train:
            dist = textdistance.hamming.normalized_similarity(pair[0], pair[1])
            x_dist[xd_index] = dist
            xd_index += 1
        
        yt_index= 0
        for pair in x_test:
            dist = textdistance.jaro_winkler(pair[0], pair[1])
            diff = 10000
            index = 0
            for train_dist in x_dist:
                if(abs(train_dist-dist) < diff):
                    diff = abs(train_dist-dist)
                    index = x_dist.index(train_dist)
            y_test[yt_index] = y_train[index]
            yt_index += 1
        
        y_test = y_test[1:]
        return y_test
   
#############################################################
        
test = corpus_classifier()
x_train, y_train = test.load_training_data()
x_val1, y_val = test.load_validation_data()

x_train_final = test.edit_distance_preprocess(x_train)

x_val_final = test.edit_distance_preprocess(x_val1)

y_test = test.jaro_winkler_classifier(x_train_final, y_train, x_val_final)
