import csv
import numpy as np
from collections import Counter
import itertools
from functools import reduce 
from porter2stemmer import Porter2Stemmer


def extract_key(v):
    return v[0]

with open('data.csv', 'w') as file_output:
    fm = ['Code','Nom']#,'Categorie','tf_idf']
    wr = csv.DictWriter(file_output,delimiter='\t', fieldnames=fm)
    wr.writeheader()
    with open('Living_Beings.csv', 'r') as file_input:
        rd = csv.reader(file_input, delimiter='\t')
    ## CSV to Array
        data = []
        for row in rd: # each row is a list
            wr.writerow({'Code': row[0],'Nom': row[1]})
            data.append(row)
   #     print data
    ## Merge the same elements in an array
        #itertools.groupby needs data to be sorted first
        data = sorted(data, key=extract_key)
        result = [[[k],[x[1] for x in g]] for k, g in itertools.groupby(data, extract_key)]
        with open("stem.txt","w") as t:
            res = []
            for i in range(0,len(result)):
                corpus = result[i][1]
                for j in range(0,len(corpus)):
                    res.append(corpus[j])
            stemmer = Porter2Stemmer()
            reslt = [c for c in res]
            reslt = [[stemmer.stem(word) for word in sentence.split(" ")] for sentence in res]
            flat_list = [item for sublist in reslt for item in sublist]
            listLines = []
            for index in flat_list:
                if index in listLines:
                    pass
                else:
                    listLines.append(index)
            print(listLines)
            print >> t,listLines
