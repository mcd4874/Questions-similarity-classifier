#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
from sklearn.metrics import log_loss
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from helper import generate_train_test_csv,load_data,generate_model,load_model,clean_data,generate_input,generate_BOW,generate_word_TI_DF,generate_n_gram_TF_IDF,generate_character_ngram_TF_IDF,generate_report
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df_train = load_data('./input/train_questions.csv')
df_valid = load_data('./input/valid_questions.csv')
# df = df_train[:100]
df = df_train
# df_valid = df_valid[:10]
df.groupby("is_duplicate")['id'].count().plot.bar()
plt.show()

# In[ ]:





# In[6]:


a = 0 
for i in range(a,a+10):
    print(df.question1[i])
    print(df.question2[i])
    print()


# In[7]:


SPECIAL_TOKENS = {
    'quoted': 'quoted_item',
    'non-ascii': 'non_ascii_word',
    'undefined': 'something'
}




# In[8]:


df = clean_data(df)
df_valid = clean_data(df_valid)
# In[9]:


a = 0 
for i in range(a,a+10):
    print(df.question1[i])
    print(df.question2[i])
    print()


# In[ ]:


# BOW +XGboost

count_vect = generate_BOW(df)
generate_model(count_vect,"BOW_context_model.pkl")
count_vect = load_model("BOW_context_model.pkl")
X_train,y_train = generate_input(count_vect,df)
X_valid, y_valid = generate_input(count_vect,df_valid)
# xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train)
xgb_model = RandomForestClassifier()
xgb_model.fit(X_train,y_train)
# generate_model(xgb_model,"gbx_BOW.pkl")
# xgb_model = load_model("gbx_BOW.pkl")
generate_model(xgb_model,"rf_BOW.pkl")
xgb_model = load_model("rf_BOW.pkl")
print("valid report ")
generate_report(xgb_model,X_valid,y_valid)
print("finish first 1")
# In[ ]:


vect_TI_DF = generate_word_TI_DF(df)
generate_model(vect_TI_DF,"word_DFTIF_context_model.pkl")
vect_TI_DF = load_model("word_DFTIF_context_model.pkl")
X_train,y_train = generate_input(vect_TI_DF,df)
X_valid, y_valid = generate_input(vect_TI_DF,df_valid)
# xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train)
xgb_model = RandomForestClassifier()
xgb_model.fit(X_train,y_train)
# generate_model(xgb_model,"gbx_word_TIDF.pkl")
# xgb_model = load_model("gbx_word_TIDF.pkl")
generate_model(xgb_model,"rf_word_TIDF.pkl")
xgb_model = load_model("rf_word_TIDF.pkl")
generate_report(xgb_model,X_valid,y_valid)
print("finish first 2")

vect_TI_DF = generate_n_gram_TF_IDF(df)
generate_model(vect_TI_DF,"word_ngram_DFTIF_context_model.pkl")
vect_TI_DF = load_model("word_ngram_DFTIF_context_model.pkl")
X_train,y_train = generate_input(vect_TI_DF,df)
X_valid, y_valid = generate_input(vect_TI_DF,df_valid)
# xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train)
xgb_model = RandomForestClassifier()
xgb_model.fit(X_train,y_train)
# generate_model(xgb_model,"gbx_ngram_TFIDF.pkl")
# xgb_model = load_model("gbx_ngram_TFIDF.pkl")
generate_model(xgb_model,"rf_ngram_TFIDF.pkl")
xgb_model = load_model("rf_ngram_TFIDF.pkl")
generate_report(xgb_model,X_valid,y_valid)
print("finish first 3")

vect_TI_DF = generate_character_ngram_TF_IDF(df)
generate_model(vect_TI_DF,"character_ngram_DFTIF_context_model.pkl")
vect_TI_DF = load_model("character_ngram_DFTIF_context_model.pkl")
X_train,y_train = generate_input(vect_TI_DF,df)
X_valid, y_valid = generate_input(vect_TI_DF,df_valid)
# xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train)
xgb_model = RandomForestClassifier()
xgb_model.fit(X_train,y_train)
# generate_model(xgb_model,"gbx_character_ngram_TFIDF.pkl")
# xgb_model = load_model("gbx_character_ngram_TFIDF.pkl")
generate_model(xgb_model,"rf_character_ngram_TFIDF.pkl")
xgb_model = load_model("rf_character_ngram_TFIDF.pkl")
generate_report(xgb_model,X_valid,y_valid)
print("finish first 4")