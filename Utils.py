


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split






df_train = pd.read_csv('./input/train_questions.csv').dropna()
# df_train,df_test = train_test_split(df,test_size=0.15, random_state=0)


# a = 0
# for i in range(a,a+10):
#     if(df.is_duplicate[i] == 1):
#         print("dup uestions")
#         print(df.question1[i])
#         print(df.question2[i])
#     else:
#         print("not dup uestions")
#         print(df.question1[i])
#         print(df.question2[i])
#     print()



# print('Total number of question pairs for training: {}'.format(len(df_train)))
# print('Duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean()*100, 2)))
# qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
# print('Total number of questions in the training data: {}'.format(len(
#     np.unique(qids))))
# print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))
#
# plt.figure(figsize=(12, 5))
# plt.hist(qids.value_counts(), bins=50)
# plt.yscale('log', nonposy='clip')
# plt.title('Log-Histogram of question appearance counts')
# plt.xlabel('Number of occurences of question')
# plt.ylabel('Number of questions')
# plt.savefig('./graph/questions-appearance.jpg')
# plt.show()
# print()
#
#
#
#
# pal = sns.color_palette()
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
# # test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)
#
# dist_train = train_qs.apply(len)
# dist_test = test_qs.apply(len)
# plt.figure(figsize=(15, 10))
# plt.hist(dist_train, bins=200, range=[0, 200], color=pal[2], normed=True, label='train')
# plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], normed=True, alpha=0.5, label='test')
# plt.title('Normalised histogram of character count in questions', fontsize=15)
# plt.legend()
# plt.xlabel('Number of characters', fontsize=15)
# plt.ylabel('Probability', fontsize=15)
# plt.savefig('./graph/character-count.jpg')
# plt.show()
# print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(),
#                           dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))
#
#
# # In[13]:
#
#
from nltk.corpus import stopwords
#
stops = set(stopwords.words("english"))
#
def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R
#
# plt.figure(figsize=(15, 5))
train_word_match = df_train.apply(word_match_share, axis=1, raw=True)
# plt.hist(train_word_match[df_train['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')
# plt.hist(train_word_match[df_train['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')
# plt.legend()
# plt.title('Label distribution over word_match_share', fontsize=15)
# plt.xlabel('word_match_share', fontsize=15)
# plt.savefig('./graph/label-distribution-word-match-share.jpg')
# plt.show()

# In[14]:


from collections import Counter

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}


# In[15]:


def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R
plt.figure(figsize=(15, 5))
tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)
plt.hist(tfidf_train_word_match[df_train['is_duplicate'] == 0].fillna(0), bins=20, normed=True, label='Not Duplicate')
plt.hist(tfidf_train_word_match[df_train['is_duplicate'] == 1].fillna(0), bins=20, normed=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Label distribution over tfidf_word_match_share', fontsize=15)
plt.xlabel('word_match_share', fontsize=15)




x_train = pd.DataFrame()
# x_test = pd.DataFrame()
x_train['word_match'] = train_word_match
x_train['tfidf_word_match'] = tfidf_train_word_match
# x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)
# x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)

y_train = df_train['is_duplicate'].values
# y_test = df_test['is_duplicate'].values


x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)


# In[ ]:


import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from helper import generate_train_test_csv,load_data,generate_model,load_model,clean_data,generate_input,generate_BOW,generate_word_TI_DF,generate_n_gram_TF_IDF,generate_character_ngram_TF_IDF,generate_report
xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(x_train, y_train)
generate_model(xgb_model,"./model/gbx_wordMatch.pkl")
xgb_model = load_model("./model/gbx_wordMatch.pkl")
print("valid report ")
generate_report(xgb_model,x_valid,y_valid)
print("finish first 1")



rf_model = RandomForestClassifier()
rf_model.fit(x_train,y_train)
# rf_model
generate_model(rf_model,"./model/rf_wordMatch.pkl")
rf_model = load_model("./model/rf_wordMatch.pkl")
print("valid report ")
generate_report(rf_model,x_valid,y_valid)
print("finish first 1")


