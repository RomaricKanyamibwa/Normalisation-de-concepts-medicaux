import csv
import numpy as np
from collections import Counter
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer

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
        print result[0]
        with open("tfidf.txt","w") as t:
            for i in range(0,len(result)):
                corpus = result[i][1]
                vectorizer = TfidfVectorizer()
                matrix = vectorizer.fit_transform(corpus)
                feature_names = vectorizer.get_feature_names()
                doc = 0
                feature_index = matrix[doc,:].nonzero()[1]
                tfidf_scores = zip(feature_index, [matrix[doc, x] for x in feature_index])
                for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
                    t.write("------- {} -------".format(result[i][0]))
                    print >> t, (w, s)
