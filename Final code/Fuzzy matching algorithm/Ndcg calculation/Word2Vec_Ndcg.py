import pandas as pd
import numpy as np
import d6tjoin.top1
import d6tjoin.utils
import jellyfish
import affinegap
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings("ignore")
from topn import *
import topn
from preprocessing import *
from sklearn.metrics.pairwise import cosine_similarity

dfj_bq= pd.read_parquet("bqDev_1_d030ed26b9-bq.parquet")
dfj_xp= pd.read_parquet("bqDev_1_d030ed26b9-xp.parquet")


b=Preprocessing2(dfj_bq,dfj_xp)
rightdata,leftdata=b.data_processing()

print(leftdata.shape) #(7926, 7)
print(rightdata.shape) #(226514, 7)

n1 = len(leftdata)
n2 = len(rightdata)
print(n1)
print(n2)

leftdata.reset_index(drop=True,inplace=True)
rightdata.reset_index(drop=True,inplace=True)

''' 
from gensim.models import Word2Vec
import gensim
from gensim.models import KeyedVectors
def train_vec(data):
    model = Word2Vec(data, epochs =10,min_count=1, window=3, vector_size=100)
    model.wv.save_word2vec_format('./word2vec.model', binary=True)
    return model
def get_vec(word):
    model = KeyedVectors.load_word2vec_format('./word2vec.model', binary=True)
    vec=model.wv[word]
    return vec

content=[]

for i in leftdata.company_name_clean.tolist():
    content.append(i.split(' '))

for i in rightdata.company_name_clean.tolist():
    content.append(i.split(' '))

model=train_vec(content)

# model.wv.save_word2vec_format("../../Word vector/word2vec_f1.txt", binary=False)
'''

import gensim
from gensim.models.keyedvectors import KeyedVectors
model = KeyedVectors.load_word2vec_format('word2vec_f1.txt')

# get all word
keys = model.index_to_key


# num_features represents the text word size
def average_word_vectors(words,model,vocabulary,num_features):
    feature_vector=np.zeros((num_features,),dtype='float64')
    nwords=0
    for word in words:
        if word in vocabulary:
            nwords=nwords+1
            feature_vector=np.add(feature_vector,model[word])
    if nwords:
        feature_vector=np.divide(feature_vector,nwords)
    return feature_vector

def averaged_word_vectorizer(corpus,model,num_features):
    #get the all vocabulary
    vocabulary=set(model.index_to_key)
    features=[average_word_vectors(tokenized_sentence,model,vocabulary,num_features) for tokenized_sentence in corpus]
    return np.array(features)


word2vec_1=averaged_word_vectorizer([i.split(' ') for i in leftdata.company_name_clean.tolist()],model,100)
word2vec_1=pd.DataFrame(word2vec_1)
word2vec_1.shape

word2vec_2=averaged_word_vectorizer([i.split(' ') for i in rightdata.company_name_clean.tolist()],model,100)
word2vec_2=pd.DataFrame(word2vec_2)
word2vec_2.shape


# Compute cos similarity between query and all document embeddings
block_size = 1000
true_list = np.zeros((block_size, n2), dtype=int)
pred_list = np.zeros((block_size, n2), dtype=float)   # change here int to float
cos_values = np.zeros((block_size, n2), dtype=float)
ndcg_list = []
block_number = int(n1/block_size) - 1
print(block_number)


## calculate ndcg score
for m in range(block_number):
    print(m)
    cos_values = cosine_similarity(word2vec_1.loc[m*block_size:(m+1)*block_size], word2vec_2)
    for i in range(block_size):
        scores = cos_values[i]
        domain_i = leftdata.iloc[m*block_size+i, 0]
        # generate ideal labels
        true_list[i, :] = np.array(rightdata['domain_name'] == domain_i, dtype=int)
        pred_list[i, :] = np.array(scores)

    block_ndcg = ndcg_score(true_list, pred_list, k=5)
    ndcg_list.append(block_ndcg)
    print(np.mean(ndcg_list))



total_ndcg = np.mean(ndcg_list)
print("NDCG:", total_ndcg)
## DCG: 0.586
