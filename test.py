import csv
import json
import numpy as np
from collections import Counter
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from functools import reduce 
import nltk
nltk.download('punkt')
from porter2stemmer import Porter2Stemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict


def extract_key(v):
    return v[0]

with open('data.csv', 'w') as file_output:
    fm = ['Code','Nom']#['Categorie','tf_idf']
    wr = csv.DictWriter(file_output,delimiter='\t', fieldnames=fm)
    wr.writeheader()
    with open('Living_Beings.csv', 'r') as file_input:
        rd = csv.reader(file_input, delimiter='\t')
    ## CSV to Array
        data = []
        for row in rd: # each row is a list
            wr.writerow({'Code': row[0],'Nom': row[1]})
            data.append(row)
    ## Merge the same elements in an array
        #itertools.groupby needs data to be sorted first
        data = sorted(data, key=extract_key)
        result = [[k,[x[1] for x in g]] for k, g in itertools.groupby(data, extract_key)]
        with open("dictionary.json","w") as t:
            reslt = []
            score = []
            for i in range(0,22326):
                corpus = result[i][1]
                res = []
                for j in range(0,len(corpus)):
                    res.append(corpus[j])
                vectorizer = TfidfVectorizer()
                matrix = vectorizer.fit_transform(res)
                feature_names = vectorizer.get_feature_names()
                stemmer = Porter2Stemmer()
                reslt.append(list([[result[i][0],stemmer.stem(word)] for word in w.split(' ')] for w in feature_names))
                score.append([m,])
            flat_list = [item for res in reslt for sublist in res for item in sublist]
            res3 = []
            res4 = []
            for i in range(0,len(flat_list)):
                corps1 = flat_list[i][0]
                corps2 = flat_list[i][1]
                for j in range(0,len(corps2)):
                    res4.append(corps2)
                    res3.append(corps1)
            d = defaultdict(list)
            for a, b in zip(res4, res3):
                if b in d[a]:
                    pass
                else:
                    d[a].append(b)
            t.write(json.dumps(d)+"\n")
