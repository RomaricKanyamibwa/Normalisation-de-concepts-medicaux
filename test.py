from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np 
from random import randint
import time
import sklearn

sLength=27751
NbClasse=5
#neeed ceux la aussi hoho
df_anatomy=pd.read_csv('~/ProjetMEdical/Anatomy.csv',names=["Code","Nom","Categorie"],sep="\t") 
df_Chemical=pd.read_csv('~/ProjetMEdical/Chemicals_and_drugs.csv',names=["Code","Nom","Categorie"],sep="\t") 
df_Devices=pd.read_csv('~/ProjetMEdical/Devices.csv',names=["Code","Nom","Categorie"],sep="\t") 
df_sign=pd.read_csv('~/ProjetMEdical/Sign_or_Symptom.csv',names=["Code","Nom","Categorie"],sep="\t") 
df_Concept=pd.read_csv('~/ProjetMEdical/Concept_and_Ideas.csv',names=["Code","Nom","Categorie"],sep="\t") 
df_anatomy_first=df_anatomy[sLength:2*sLength]
df_sign_first=df_sign[0:sLength]
df_Chemical_first=df_Chemical[2*sLength:3*sLength]
df_Devices_first=df_Devices[2*sLength:3*sLength]
df_Concept_first=df_Concept[3*sLength:4*sLength]

df_anatomy_first['Origine']=pd.Series(np.ones(sLength,dtype=int)*0,index=df_anatomy_first.index)
df_Chemical_first['Origine']=pd.Series(np.ones(sLength,dtype=int)*1,index=df_Chemical_first.index)
df_Devices_first['Origine']=pd.Series(np.ones(sLength,dtype=int)*2,index=df_Devices_first.index)
df_sign_first['Origine']=pd.Series(np.ones(sLength,dtype=int)*3,index=df_sign_first.index)
df_Concept_first['Origine']=pd.Series(np.ones(sLength,dtype=int)*4,index=df_Concept_first.index)
df=pd.concat([df_anatomy_first,df_Chemical_first,df_Devices_first,df_sign_first,df_Concept_first],ignore_index=True)

"""
def ListeVoisin(indicecalcul):
	for i in range(1,50252):
	if sklearn.metrics.pairwise.euclidean_distances(features[indicecalcul],features[i])<minimal and indicecalcul!=i:
		print("I dont wann wake up")
		minimal=sklearn.metrics.pairwise.euclidean_distances(features[indicecalcul],features[i])
		indice=i
"""
#df=pd.read_csv('classif.csv',names=["Code","Nom","Categorie"],sep=",") 
#df.Code[79000:]=[0] * (len(df.Code)-23000)


tfidf = TfidfVectorizer()
df.Origine=df.Origine.astype(int)
features =tfidf.fit_transform(df.Nom.values.astype(str))

X_train, X_test, y_train, y_test = train_test_split(features,df.Origine,test_size=0.05, random_state=42)

clf=LogisticRegression()
clf.fit(X_train,y_train)

#clf.sparsify()
#clf_label_i = LogisticRegression(penalty='l1').fit(X_train, y_train.toarray()).sparsify()
predictions = clf.predict(X_test)
"""
print(confusion_matrix(y_test,predictions))
liste=[]
print(classification_report(y_test,predictions))
indicetrouve=0
indicecalcul=randint(0,NbClasse*sLength)
minimal=10
for i in range(0,NbClasse*sLength):
		if df.Origine[i]==df.Origine[indicecalcul]:
			distance=sklearn.metrics.pairwise.euclidean_distances(features[indicecalcul],features[i])
			if distance<minimal and indicecalcul!=i:
				minimal=distance
				indicetrouve=i
	result=df.Code[indicecalcul]==df.Code[indicetrouve]
	liste.append(result)

"""
"""
Probleme sur le tfidf valeurs trop proches 
Peut pas comparer mot par mot 
Stemming
Calculer distance a la volee et premier filtre 
Regler tfidf ou changer de methode
tfidf globale
embeddings avec des distances 
de chquae mot
moyenne pour chaque terme 
En amont -> classification et combinaison 
Word 2 vec  permet d'avoir les vecteurs

Probleme des BroadMannArea

Numero 0 -> Devices
Numero 1 -> Living Beings
Numero 2 -> Chemicals and Drugs
Numero 3 ->  Anatomy
Numero 4 -> Conce

predict output word -> pas mal 
most similar
most similar cos 
from word embeddings to document distances


cnn -> essayer d'apprendre que deux dsynonymes sont proches etque deux non synonymes sont loins -> TRIPLET LOSS 
Regardez seq2seq
"""