# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sys import stdout
import time

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers.advanced_activations import LeakyReLU 
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import text_to_word_sequence,Tokenizer
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.utils import shuffle



tags = ["UID","Numero","Tokens","label"]

file = pd.read_csv("data.csv", names = tags)
lesfichier  = ["Anatomy","Biological_Process_or_Function","Chemicals_and_drugs","Concept_and_Ideas","Devices","Disorders","Genes_and_Proteins","Medical_procedures","Sign_or_Symptom"]

#file = file.drop(0)
file = file.drop(columns="UID")

adrop = []


for i,nom in enumerate(file["Tokens"]):
	if(type(nom) is not str):
		adrop.append(i)

file=file.drop(adrop)


file = shuffle(file)
filetest=file[:int(round(file.shape[0]*0.01))]
filetest.shape

train_size = int(filetest.shape[0] * .8)

train_posts=filetest["Tokens"][:train_size]
train_label=filetest["label"][:train_size]
test_posts = filetest["Tokens"][train_size:]
test_label=filetest["label"][train_size:]


num_labels = 9
vocab_size = 15000
batch_size = 100

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_posts)

#MLP

x_train = tokenizer.texts_to_matrix(train_posts, mode='tfidf')
x_test = tokenizer.texts_to_matrix(test_posts, mode='tfidf')

encoder = LabelBinarizer()
encoder.fit(train_label)

y_train = encoder.transform(train_label)
y_test = encoder.transform(test_label)

model = Sequential()
model.add(Dense(1000, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
history = model.fit(x_train, y_train,batch_size=batch_size,epochs=5,verbose=1,validation_split=0.1)
# list all data in history



# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
score, acc = model.evaluate(x_test, y_test,batch_size=batch_size, verbose=1)
 
print('Test accuracy:', acc)
