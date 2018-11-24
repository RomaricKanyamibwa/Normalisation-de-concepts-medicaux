#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 21:52:34 2018

@author: nightrider
"""
import pandas as pd

#%%
df_disorders=pd.read_csv("Disorders.csv",names=["Code","Disorder","GenDisorder"],sep="\t") 

#%%

Codes=df_disorders['Code']#Y vector (outputs)
Disorders=df_disorders['Disorder']#X vector(test)
GenDisorders=df_disorders['GenDisorder']#X vector (train)

#%%
Disorders_corpus=Disorders.str.split()
GenDisorders_corpus=GenDisorders.str.split()


def create_corpus(Series_Corpus):
    #this function can be multithreaded
    Corpus=[Series_Corpus]
    for i in (Series_Corpus):
        Corpus+=i
        
    return Corpus