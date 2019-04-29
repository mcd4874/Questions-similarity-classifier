
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

def generate_train_test_csv(df):
    df,df_valid= train_test_split(df ,test_size = 0.15, random_state = 0)
    df_valid.to_csv('./input/valid_questions.csv')
    df.to_csv('./input/train_questions.csv')

def load_data(fileName):
    df = pd.read_csv(fileName).dropna()
    df = df.dropna(how="any").reset_index(drop=True)
    return df
def generate_model(model,savePath):
    with open(savePath, 'wb') as file:
        pickle.dump(model, file, -1)
    return model
def load_model(loadPath):
    with open(loadPath, 'rb') as file:
        model = pickle.load(file)
    return model

SPECIAL_TOKENS = {
    'quoted': 'quoted_item',
    'non-ascii': 'non_ascii_word',
    'undefined': 'something'
}
def clean(text, stem_words=True):
    import re
    from string import punctuation
    from nltk.stem import SnowballStemmer
    from nltk.corpus import stopwords

    def pad_str(s):
        return ' ' + s + ' '

    # if pd.isnull(text):
    #     return ''

    #    stops = set(stopwords.words("english"))
    # Clean the text, with the option to stem words.

    # Empty question

    if type(text) != str or text == '':
        return ''

    # Clean the text
    text = re.sub("\'s", " ",
                  text)  # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)
    text = re.sub("(\d+)(kK)", " \g<1>000 ", text)
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    text = re.sub("[c-fC-F]\:\/", " disk ", text)

    # remove comma between numbers, i.e. 15,000 -> 15000

    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)

    #     # all numbers should separate from words, this is too aggressive

    #     def pad_number(pattern):
    #         matched_string = pattern.group(0)
    #         return pad_str(matched_string)
    #     text = re.sub('[0-9]+', pad_number, text)

    # add padding to punctuations and special chars, we still need them later

    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)

    #    def pad_pattern(pattern):
    #        matched_string = pattern.group(0)
    #       return pad_str(matched_string)
    #    text = re.sub('[\!\?\@\^\+\*\/\,\~\|\`\=\:\;\.\#\\\]', pad_pattern, text)

    text = re.sub('[^\x00-\x7F]+', pad_str(SPECIAL_TOKENS['non-ascii']),
                  text)  # replace non-ascii word with special word

    # indian dollar

    text = re.sub("(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)

    # clean text rules get from : https://www.kaggle.com/currie32/the-importance-of-cleaning-text
    text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)
    text = re.sub(r" UK ", " England ", text, flags=re.IGNORECASE)
    text = re.sub(r" india ", " India ", text)
    text = re.sub(r" switzerland ", " Switzerland ", text)
    text = re.sub(r" china ", " China ", text)
    text = re.sub(r" chinese ", " Chinese ", text)
    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
    text = re.sub(r" quora ", " Quora ", text, flags=re.IGNORECASE)
    text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)
    text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE)
    text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
    text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
    text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE)
    text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
    text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
    text = re.sub(r" \0rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
    text = re.sub(r" ios ", " operating system ", text, flags=re.IGNORECASE)
    text = re.sub(r" gps ", " GPS ", text, flags=re.IGNORECASE)
    text = re.sub(r" gst ", " GST ", text, flags=re.IGNORECASE)
    text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
    text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
    text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)
    text = re.sub(r" III ", " 3 ", text)
    text = re.sub(r" banglore ", " Banglore ", text, flags=re.IGNORECASE)
    text = re.sub(r" J K ", " JK ", text, flags=re.IGNORECASE)
    text = re.sub(r" J\.K\. ", " JK ", text, flags=re.IGNORECASE)

    # replace the float numbers with a random number, it will be parsed as number afterward, and also been replaced with word "number"

    text = re.sub('[0-9]+\.[0-9]+', " 87 ", text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation]).lower()
    # Return a list of words
    return text

def clean_data(df):
    df['question1'] = df['question1'].apply(clean)
    df['question2'] = df['question2'].apply(clean)
    return df

def generate_input(count_vect,df):
    trainq1_trans = count_vect.transform(df['question1'].values)
    trainq2_trans = count_vect.transform(df['question2'].values)
    labels = df['is_duplicate'].values
    X = scipy.sparse.hstack((trainq1_trans,trainq2_trans))
    y = labels
    return [X,y]

def generate_BOW(df):
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(pd.concat((df['question1'],df['question2'])).unique())
    return count_vect
def generate_word_TI_DF(df):
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(pd.concat((df['question1'], df['question2'])).unique())
    return tfidf_vect

def generate_n_gram_TF_IDF(df):
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)
    tfidf_vect_ngram.fit(pd.concat((df['question1'], df['question2'])).unique())
    return tfidf_vect_ngram

def generate_character_ngram_TF_IDF(df):
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(pd.concat((df['question1'],df['question2'])).unique())
    return tfidf_vect_ngram_chars

def generate_report(model,X_valid,y_valid):
    xgb_prediction = model.predict(X_valid)
    print('validation F1 -  test:', f1_score(y_valid, model.predict(X_valid), average='macro'))
    print(classification_report(y_valid, xgb_prediction))

# def build_model(df,df_valid,count_vect,xgb_model,):
#     X_train, y_train = generate_input(count_vect, df)
#     X_valid, y_valid = generate_input(count_vect, df_valid)
#     # xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train, y_train)
#     # xgb_model = RandomForestClassifier()
#     xgb_model
#     generate_model(xgb_model, "gbx_BOW.pkl")
#     xgb_model = load_model("gbx_BOW.pkl")
#     print("valid report ")
#     generate_report(xgb_model, X_valid, y_valid)