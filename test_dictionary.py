import json
import numpy as np
from collections import Counter
import itertools
from functools import reduce 
from porter2stemmer import Porter2Stemmer
from collections import defaultdict
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

with open('tfidf6.json', 'r') as file:
	data = json.loads(file.read())
	strg = 'cuban treefrog'
	words = word_tokenize(strg)
	print(words)
	ps = Porter2Stemmer()
	res = []
	s = []
	for w in words:
		res.append(ps.stem(w))
	for i in res:
		for key, value in data.items():
			if(i == key):
				if type(value) is list:
					for j in value:
						s.append(j)
				else:
					s.append(value)
	print(s)

#Take the occurence maximum
code = Counter(s)
print(code)
