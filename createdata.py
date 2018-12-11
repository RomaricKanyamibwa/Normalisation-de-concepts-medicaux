# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import text_to_word_sequence,Tokenizer



listfichier  = ["Anatomy","Biological_Process_or_Function","Chemicals_and_drugs","Concept_and_Ideas","Devices","Disorders","Genes_and_Proteins","Medical_procedures","Sign_or_Symptom"]



source = "/home/kotama/Bureau/Nlp_projet/Data/tokenized/"
extension = ".csv"

file = open("data.csv", "w")
 
sizetmp =[]
for fichier in listfichier:
	print(fichier)
	chaine = source+fichier+extension
	fi = pd.read_csv(chaine, sep = '\t', header=None)
	fi[2]=fichier
	fi.to_csv(file)

"""
depart = 0
tmp=0

for i,nom in enumerate(listfichier):
	tmp = tmp+sizetmp[i]
	for j in range(depart,tmp):
		fi[2][j]=nom
	depart = depart+tmp

file.close()

size = fi.shape[0]
dataframe = []

for i in range(size):
	xx = text_to_word_sequence(fi[1][i])
	yy = listfichier.index(fi[2][i])
	dataframe.append([xx,yy])


print(dataframe.shape)
"""