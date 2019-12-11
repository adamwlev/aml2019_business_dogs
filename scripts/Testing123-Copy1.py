import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import pairwise_distances

import nltk
import string
import os
from collections import Counter

import tensorflow_hub as hub
import gensim


### Description for train data
desc_files = len(os.listdir('../descriptions_train'))
all_desc_train = []

for i in range(desc_files):
    empty_str = ''
    for line in open(f'../descriptions_train/{i}.txt'):
        empty_str += line.replace('\n',' ')
    all_desc_train.append(empty_str)


### Tags for train data
tag_files = len(os.listdir('../tags_train'))
all_tags_train = []

for i in range(tag_files):
    nouns = ''
    for line in open(f'../tags_train/{i}.txt'):
        nouns += line.replace(':',' ')
    all_tags_train.append(nouns.replace('\n', ' '))


### Description for test data
desc_files = len(os.listdir('../descriptions_test'))
all_desc_test = []

for i in range(desc_files):
    empty_str = ''
    for line in open(f'../descriptions_test/{i}.txt'):
        empty_str += line.replace('\n',' ')
    all_desc_test.append(empty_str)



### Tags for test data
tag_files = len(os.listdir('../tags_test'))
all_tags_test = []

for i in range(tag_files):
    nouns = ''
    for line in open(f'../tags_test/{i}.txt'):
        nouns += line.replace(':',' ')
    all_tags_test.append(nouns.replace('\n', ' '))



all_docs = []
all_docs.extend(all_desc_train)
all_docs.extend(all_desc_test)
all_docs.extend(all_tags_train)
all_docs.extend(all_tags_test)


train_1000 = pd.read_csv('../features_train/features_resnet1000_train.csv', header=None)
train_2048 = pd.read_csv('../features_train/features_resnet1000intermediate_train.csv', header=None)
test_1000 = pd.read_csv('../features_test/features_resnet1000_test.csv', header=None)
test_2048 = pd.read_csv('../features_test/features_resnet1000intermediate_test.csv', header=None)



def get_num(string):
    string = string.replace('.', ' ').replace('/', ' ')
    num = [int(s) for s in string.split() if s.isdigit()]
    return num[0]

def parse_to_numpy(pd):
    images_idx = []
    for string in pd[0]:
        images_idx.append(get_num(string))

    pd.insert(1, "Image_Index", images_idx, True)
    pd = pd.sort_values(by=['Image_Index'])
    pd = pd.reset_index(drop=True)
    del pd['Image_Index']
    del pd[0]
    np = pd.to_numpy()
    return np


train_1000 = parse_to_numpy(train_1000)
train_2048 = parse_to_numpy(train_2048)
test_1000 = parse_to_numpy(test_1000)
test_2048 = parse_to_numpy(test_2048)


### Google embedding
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

train_desc_tf = embed(all_desc_train).numpy()
test_desc_tf = embed(all_desc_test).numpy()
train_tags_tf = embed(all_tags_train).numpy()
test_tags_tf = embed(all_tags_test).numpy()


## tfidf
stop_words = set(nltk.corpus.stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=2)
vectorizer.fit(all_desc_train+all_tags_train)

train_desc_bow = np.array(vectorizer.transform(all_desc_train).todense())
test_desc_bow = np.array(vectorizer.transform(all_desc_test).todense())
train_tags_bow = np.array(vectorizer.transform(all_tags_train).todense())
test_tags_bow = np.array(vectorizer.transform(all_tags_test).todense())


# word2vec
embed = hub.load("https://tfhub.dev/google/Wiki-words-250/2")


punct = set(string.punctuation)
def process_doc(doc):
    doc = doc.lower()
    doc = ''.join(c for c in doc if c not in punct)
    doc = doc.split()
    doc = [word for word in doc if word not in stop_words]
    return doc

gensim_docs = [process_doc(doc) for doc in all_docs]
w2v_vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=1)
w2v_vectorizer.fit([' '.join(doc) for doc in gensim_docs]);


word2vec_train_desc = np.zeros((10000,250))
for i in range(10000):
    with open(f'../descriptions_train/{i}.txt') as f:
        text = f.read()
        words = process_doc(text)
        num_words = len(words)
        word_counter = Counter(words)
        total = 0
        for word in set(words):
            if word not in w2v_vectorizer.vocabulary_:
                continue
            index = w2v_vectorizer.vocabulary_[word]
            weight = word_counter[word]*w2v_vectorizer.idf_[index] ## tfidf weight
            word2vec_train_desc[i] += weight*embed([word])
            total += weight
        word2vec_train_desc[i] /= total

word2vec_test_desc = np.zeros((2000,250))
for i in range(2000):
    with open(f'../descriptions_test/{i}.txt') as f:
        text = f.read()
        words = process_doc(text)
        num_words = len(words)
        word_counter = Counter(words)
        total = 0
        for word in set(words):
            if word not in w2v_vectorizer.vocabulary_:
                continue
            index = w2v_vectorizer.vocabulary_[word]
            weight = word_counter[word]*w2v_vectorizer.idf_[index] ## tfidf weight
            word2vec_test_desc[i] += weight*embed([word])
            total += weight
        word2vec_test_desc[i] /= total


word2vec_train_tags = np.zeros((10000,250))
for i in range(10000):
    with open(f'../tags_train/{i}.txt') as f:
        text = f.read()
        words = process_doc(text)
        num_words = len(words)
        word_counter = Counter(words)
        total = 0
        for word in set(words):
            if word not in w2v_vectorizer.vocabulary_:
                continue
            index = w2v_vectorizer.vocabulary_[word]
            weight = word_counter[word]*w2v_vectorizer.idf_[index] ## tfidf weight
            word2vec_train_tags[i] += weight*embed([word])
            total += weight
        if total!=0:
            word2vec_train_tags[i] /= total

word2vec_test_tags = np.zeros((2000,250))
for i in range(2000):
    with open(f'../tags_test/{i}.txt') as f:
        text = f.read()
        words = process_doc(text)
        num_words = len(words)
        word_counter = Counter(words)
        total = 0
        for word in set(words):
            if word not in w2v_vectorizer.vocabulary_:
                continue
            index = w2v_vectorizer.vocabulary_[word]
            weight = word_counter[word]*w2v_vectorizer.idf_[index] ## tfidf weight
            word2vec_test_tags[i] += weight*embed([word])
            total += weight
        if total!=0:
            word2vec_test_tags[i] /= total


def get_prediction(vecs,pics):
    dists = pairwise_distances(vecs,pics,metric='cosine')
    return dists.T.argsort(1)
def map_20(ranks):
    return np.mean([(20-rank)/20 if rank<20 else 0 for rank in ranks])
def map_20_2(ranks):
    return np.mean([1/(1+rank) if rank<20 else 0 for rank in ranks])
def evaluate(vectors,label_vectors):
    preds = get_prediction(vectors,label_vectors)
    ranks = [np.argwhere(vec==i)[0][0] for i,vec in enumerate(preds)]
    return np.mean(ranks),map_20(ranks),map_20_2(ranks)
def get_top_20(descr_id):
    return preds[descr_id][:20]
def save_submission():
    data = []
    for i in range(2000):
        data.append(['%d.txt' % (i,),' '.join('%d.jpg' % (pic_id,) for pic_id in get_top_20(i))])
    pd.DataFrame(data,columns=['Descritpion_ID','Top_20_Image_IDs']).to_csv('submission.csv',index=False)


# TF Embeddings + TFIDF

train_desc = np.hstack((train_desc_tf, train_desc_bow))
#test_desc = np.hstack((test_desc_tf, test_desc_bow))

train_pic = np.hstack((train_1000, train_tags_tf, train_tags_bow))
#test_pic = np.hstack((test_1000, test_tags_tf, test_tags_bow))

print(train_pic.shape, train_desc.shape)

kf = KFold(n_splits=5)
rcv = RidgeCV(alphas=np.linspace(1,40,20))
for train_index, test_index in kf.split(train_pic):
    rcv.fit(train_pic[train_index], train_desc[train_index])
    pred = rcv.predict(train_pic[test_index])
    output = evaluate(pred, train_desc[test_index])
    print(output)


# TF Embeddings + Word2Vec


train_desc = np.hstack((train_desc_tf, word2vec_train_desc))
#test_desc = np.hstack((test_desc_tf, word2vec_test_desc))

train_pic = np.hstack((train_1000, train_tags_tf, word2vec_train_tags))
#test_pic = np.hstack((test_1000, test_tags_tf, word2vec_test_tags))

print(train_pic.shape, train_desc.shape)

kf = KFold(n_splits=5)
rcv = RidgeCV(alphas=np.linspace(1,40,20))
for train_index, test_index in kf.split(train_pic):
    rcv.fit(train_pic[train_index], train_desc[train_index])
    pred = rcv.predict(train_pic[test_index])
    output = evaluate(pred, train_desc[test_index])
    print(output)


import lightgbm as lgbm
import itertools

reg = lgbm.LGBMRegressor(n_estimators=300,learning_rate=.05,num_leaves=27,reg_lambda=.1,colsample_bytree=.9)
print('booster:')
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(train_pic):
    preds = np.zeros((2000,train_desc.shape[1]))
    for i in range(train_desc.shape[1]):
        if (i%25)==0: print(i)
        reg.fit(train_pic[train_index], np.ravel(train_desc[train_index,i:i+1]))
        pred = reg.predict(train_pic[test_index])
        preds[:,i] = np.ravel(pred)
    output = evaluate(preds, train_desc[test_index])
    print(output)

