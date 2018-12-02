from gensim.models import Word2Vec
import re
from random import randint
import time
from sklearn.neighbors import NearestNeighbors
def to_list(text):
	return re.sub("[^\w]", " ", text).split()

mystr = 'This is a string, with words!'
wordList = re.sub("[^\w]", " ",  mystr).split()
#Need ce fichier ici 
model = Word2Vec.load("sg-s0200-w08-m004-ns05-s0.001-a0.025-i05.pkl")
execfile("../test.py")
L1=[]
L2=[]
#Need ceux la
df_Chemical=pd.read_csv('~/ProjetMEdical/Chemicals_and_drugs.csv',names=["Code","Nom","Categorie"],sep="\t") 
df_Devices=pd.read_csv('~/ProjetMEdical/Devices.csv',names=["Code","Nom","Categorie"],sep="\t") 
df_sign=pd.read_csv('~/ProjetMEdical/Sign_or_Symptom.csv',names=["Code","Nom","Categorie"],sep="\t") 
df_Concept=pd.read_csv('~/ProjetMEdical/Concept_and_Ideas.csv',names=["Code","Nom","Categorie"],sep="\t") 
ListeTheme=["anatomy","device","sign","symptom","concept","ideas"]
""""idtkt=randint(0,len(df))
wordList=re.sub("[^\w]", " ",  df.iloc[idtkt]['Categorie']).split()
maxS=0
for theme in ListeTheme:
	newval=model.n_similarity(wordList,theme)
	if newval>maxS:
		maxS=newval
		themefinal=theme
"""
df.dropna()

newlist=tfidf.transform(["superior part of head","toothache"])

ptest={k: (g['Origine'],g['Nom'].tolist()) for k,g in df.groupby('Code')}
ptest2={k: (g['Origine'],g['Nom'].tolist()) for k,g in df_sign_first.groupby('Code')}

text="toothache"
prediction=clf.predict(newlist)
label=prediction[1]
wordtext=to_list(text)
maxVal=0
val=True
tzrro=time.time()
"""
for key, value in ptest2.iteritems():
	for i in value[1]:
		val=True
		if type(i) is str:
			wordList1=to_list(i)
			if (not wordList1)==False:
					for k in wordList1:
						if(k in model.wv.vocab)==False:
							val=False


					if val==True:
						newvall=model.n_similarity(wordList1,wordtext)
						if newvall>maxVal:
							maxVal=newvall
							realkey=key
print (time.time()-tzrro)
"""

dicoIndiceKey={}
compteur=0
ExpressionList=[]
for key, value in ptest.iteritems():
	for i in value[1]:
		val=True
		if type(i) is str:
			wordList1=to_list(i)
			if (not wordList1)==False:
				listeVector=[]
				for k in wordList1:
					if(k in model.wv.vocab)==False:
						val=False
					else:
						listeVector.append(model.wv[k])

				if val==True:
						VectorMean=sum(listeVector)/len(listeVector)
						ExpressionList.append(VectorMean)
						dicoIndiceKey[compteur]=key
						compteur=compteur+1

neigh=NearestNeighbors(3)

neigh.fit(ExpressionList)

ExpressionTest=[]
ExpressionTest.append(model.wv["superior"])
ExpressionTest.append(model.wv["part"])
ExpressionTest.append(model.wv["of"])
ExpressionTest.append(model.wv["head"])
VectorTest=sum(ExpressionTest)/len(ExpressionTest)
prediction=neigh.kneighbors([model.wv["toothache"],VectorTest],return_distance=False)
print (ptest[dicoIndiceKey[prediction[0][0]]])
print (ptest[dicoIndiceKey[prediction[1][0]]])

"""
OKay J'ai mes expressions
Je calcule les tfidf
je predis le concept
Le concept en main je recupere les mots les plus proches


librarie qui fait le plus proche voisin au lieu de le calculer en dure, ca le fout dans l'espace -> 
faire des super tests d'evaluations /data visualisation tizni/tisni/tease ni 
Comme c 'est de l'abalyse contextuel chercher la simalarite c "est de la merde
"""