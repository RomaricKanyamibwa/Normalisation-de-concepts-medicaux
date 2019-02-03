import csv
import numpy as np
from collections import Counter
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer

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
        result = [[[k],[x[1] for x in g]] for k, g in itertools.groupby(data, extract_key)]
        with open("tfidf.txt","w") as t:
            res = []
            for i in range(0,len(result)):
                corpus = result[i][1]
                for j in range(0,len(corpus)):
                    res.append(corpus[j])
                vectorizer = TfidfVectorizer()
                matrix = vectorizer.fit_transform(res)
            ##Here the tuple represents, document no. (in this case sentence no.) and feature no.
            t.write(" ------- Tf-idf -------\n")
            print >> t, matrix
            t.write("\n\n\n ------- List of features -------\n")
            ##List of features
            for i, feature in enumerate(vectorizer.get_feature_names()):
                print >> t, (i,feature)
