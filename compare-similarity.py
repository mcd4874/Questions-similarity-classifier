#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,log_loss, accuracy_score


import gensim

import nltk

# from xgboost import XGBClassifier


# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import os
print(os.listdir("./input"))

# Any results you write to the current directory are saved as output.

# use 


# In[2]:


df = pd.read_csv('./input/questions.csv').dropna()


# In[3]:



df_train,df_test = train_test_split(df,test_size=0.2, random_state=42)
print(df_train.shape)


# In[4]:


def calculate_cosim(row):
    a = cv.transform([row.question1])
    b = cv.transform([row.question2])
    return cosine_similarity(a,b).ravel()[0]
def makePredict(row):
    if (row.cosim > 0.5):
        return 1
    return 0
def generatePrediction(df):
    df['cosim'] =  df.apply (lambda row: calculate_cosim(row), axis=1)
    df['predict'] =  df.apply (lambda row: makePredict(row), axis=1)
    return df

def evaluate_acc(actual,predict):
    return accuracy_score(actual, predict)
    
# df_train['cosim'] =  df_train.apply (lambda row: calculate_cosim(row), axis=1)
# df_test['cosim'] =  df_test.apply (lambda row: calculate_cosim(row), axis=1)
# df_train['predict'] =  df_train.apply (lambda row: makePredict(row), axis=1)
# df_test['predict'] =  df_test.apply (lambda row: makePredict(row), axis=1)


# In[5]:


# accuracy_score(df_train.is_duplicate, df_train.predict)
# df_test.head()
# accuracy_score(df_test.is_duplicate, df_test.predict)


# we start to train word embedding
# 

# In[6]:


class MySentences(object):
    """MySentences is a generator to produce a list of tokenized sentences 
    
    Takes a list of numpy arrays containing documents.
    
    Args:
        arrays: List of arrays, where each element in the array contains a document.
    """
    def __init__(self, *arrays):
        self.arrays = arrays
 
    def __iter__(self):
        for array in self.arrays:
            for document in array:
                for sent in nltk.sent_tokenize(document):
                    yield nltk.word_tokenize(sent)

def get_word2vec(sentences, location,retrain):
    """Returns trained word2vec
    
    Args:
        sentences: iterator for sentences
        
        location (str): Path to save/load word2vec
    """
    if os.path.exists(location) and not retrain :
        print('Found {}'.format(location))
        model = gensim.models.Word2Vec.load(location)
        return model
    
    print('{} not found. training model'.format(location))
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    print('Model done training. Saving to disk')
    model.save(location)
    return model


# In[7]:


print(df_train['question1'].shape)
corpus_train = np.concatenate((df_train['question1'], df_train['question2']), axis=0)
print(corpus_train.shape)
w2vec = get_word2vec(
    MySentences(
       corpus_train
    ),
    'w2vmodel.model',
    False
)


# In[8]:



class MyTokenizer:
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        transformed_X = []
        for document in X:
            tokenized_doc = []
            for sent in nltk.sent_tokenize(document):
                tokenized_doc += nltk.word_tokenize(sent)
            transformed_X.append(np.array(tokenized_doc))
        return np.array(transformed_X)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.wv.syn0[0])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = MyTokenizer().fit_transform(X)
        
        return np.array([
            np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
    def fit_transform(self, X, y=None):
        return self.transform(X)


# In[9]:


# generate embedding vector

mean_embedding_vectorizer = MeanEmbeddingVectorizer(w2vec)

question1_vectors = mean_embedding_vectorizer.fit_transform(df_train['question1'])
question2_vectors = mean_embedding_vectorizer.fit_transform(df_train['question2'])

question1_test_vectors = mean_embedding_vectorizer.fit_transform(df_test['question1'])
question2_test_vectors = mean_embedding_vectorizer.fit_transform(df_test['question2'])
print(question1_vectors.shape)


# In[38]:


# util function to generate features
def sqrt_sum(a,b):
    return np.sqrt(np.sum((a-b)**2, axis=1))


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    if (np.linalg.norm(vector) == 0):
        return np.zeros(vector.shape)
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    if (v1_u.all() == 0 or v2_u.all() == 0):
        return 0
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
def angle_wise_matrix(matrix1, matrix2):
    size = matrix1.shape[0]
    result = np.zeros(size)
    for i in range (size):
        result[i] = angle_between(matrix1[i],matrix2[i])
    return result
def calculate_cosim_vectors(matrix1,matrix2):
    result = np.zeros(matrix1.shape[0])
    for i in range(matrix1.shape[0]):        
        result[i] = cosine_similarity(matrix1[i:i+1],matrix2[i:i+1])
    return result



# In[39]:


train_cosim = calculate_cosim_vectors(question1_vectors,question2_vectors)
test_cosim = calculate_cosim_vectors(question1_test_vectors,question2_test_vectors)


# In[12]:



dist = sqrt_sum(question1_vectors,question2_vectors)
angles = angle_wise_matrix(question1_vectors,question2_vectors)
dist_test = sqrt_sum(question1_test_vectors,question2_test_vectors)
angles_test = angle_wise_matrix(question1_test_vectors,question2_test_vectors)


# In[13]:


print(question1_vectors.shape)
vec_representation = np.concatenate([question1_vectors, question2_vectors],axis=1)
vec_representation_test = np.concatenate([question1_test_vectors, question2_test_vectors],axis=1)
print(vec_representation.shape)


# In[47]:


# print("nan in dist : ",np.isnan(dist).any())
# print("nan in angle : ",np.isnan(angles).any())

# print("nan in test dist : ",np.isnan(dist_test).any())
# print("nan in test angle : ",np.isnan(angles_test).any())
# x_input =  np.stack([dist, angles,train_cosim],axis=1)
# x_input =  np.stack([dist, train_cosim],axis=1)
x_input =  np.stack([dist, angles],axis=1)



y_input = df_train['is_duplicate'].values

# x_test =  np.stack([dist_test, angles_test,test_cosim],axis=1)
# x_test =  np.stack([dist_test, test_cosim],axis=1)
x_test =  np.stack([dist_test, angles_test],axis=1)



y_test = df_test['is_duplicate'] .values


size = len(y_input)
# y_input= y_input.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
print(x_input.shape)
print(x_test.shape)
print(y_input.shape)


# In[48]:


import pickle
def generate_model(x_input,y_input,model,savePath):
    model.fit(x_input,y_input)
    with open(savePath, 'wb') as file:
        pickle.dump(model, file, -1)
    return model
def load_model(loadPath):
    with open(loadPath, 'rb') as file:
        model = pickle.load(file)
    return model
def get_report(x_test,y_test,model):
    predict = model.predict(x_test)
    print(classification_report(y_test, predict))


# In[49]:


clf = SVC(gamma='auto')
# rf= RandomForestClassifier()


# In[50]:


# model = generate_model(x_input,y_input,rf,"rf.pkl")
# model = generate_model(vec_representation,y_input,rf,"rf.pkl")
model = generate_model(vec_representation,y_input,clf,"svm.pkl")


# In[51]:


# get_report(x_test,y_test,model)
# get_report(vec_representation_test,y_test,model)
get_report(vec_representation_test,y_test,model)


# In[52]:


# def generate_time_step_data(input,timeStep):
    


# In[53]:


# transform shape for LSTM (require 3D array)
# time_step = 1
# input_features = x_input.shape[1]
# # output_festures = y_input.shape[1]
# # print(input_features)
# # print(output_festures)
#
#
# x_input = x_input.reshape(int(x_input.shape[0]),input_features,1)
# x_test = x_test.reshape(int(x_test.shape[0]),input_features,1)
#
# # y_input = y_input.reshape(int(y_input.shape[0]),time_step,output_festures)
# # y_test = y_test.reshape(int(y_test.shape[0]),time_step,output_festures)
#
# print(x_input.shape, y_input.shape, x_test.shape, y_test.shape)
#
#
# # In[61]:
#
#
# from keras.layers import Dropout
# def create_model(n_batch,features,time_step = 2):
#     print(features)
#     print(time_step)
#     model = Sequential()
# #     model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
#     model.add(LSTM(100,input_shape=(time_step,features)))
#     model.add(Dense(256,name='FC1', activation = 'relu'))
# #     model.add(Dropout(0.2))
#     model.add(Dense(1,name='out_layer', activation='sigmoid'))
#     return model
#
#
# # In[63]:
#
#
# model = create_model(x_input.shape[0],1, 2)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
#
#
# # In[ ]:
#
#
# model.fit(x_input,y_input,batch_size=128,epochs=154,
#           validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
#
#
# # In[ ]:
#
#
# accr = model.evaluate(x_test,y_test)
# print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


# In[ ]:





# In[ ]:




