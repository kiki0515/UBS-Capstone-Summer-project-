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
from sentence_transformers import SentenceTransformer, util

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

# Part 1 Initial Run

query = list(leftdata["company_name_clean"])
docs = list(rightdata["company_name_clean"])

# Load the model
model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L6-cos-v5')

# Encode query and documents
query_emb = model.encode(query)
doc_emb = model.encode(docs)


print(len(query))
print(query_emb.shape)

print(len(docs))
print(doc_emb.shape)

# Compute cos similarity between query and all document embeddings
block_size = 1000
true_list = np.zeros((block_size, n2), dtype=int)
pred_list = np.zeros((block_size, n2), dtype=float)   # change here
cos_values = np.zeros((block_size, n2), dtype=float)
ndcg_list = []
block_number = int(n1/block_size) - 1
print(block_number)

# calculate ndcg score

for m in range(block_number):
    print(m)
    cos_values = cosine_similarity(query_emb[m*block_size:(m+1)*block_size], doc_emb)
    for i in range(block_size):
        scores = cos_values[i]
        domain_i = leftdata.iloc[m*block_size+i, 0]
        # generate ideal labels
        true_list[i, :] = np.array(rightdata['domain_name'] == domain_i, dtype=int) # if match : 1, not match: give 0    
        pred_list[i, :] = np.array(scores)                          # keep cos similarity values float. in order to rank 5 more accurately
    block_ndcg = ndcg_score(true_list, pred_list, k=5)
    ndcg_list.append(block_ndcg)
    print(np.mean(ndcg_list))

total_ndcg = np.mean(ndcg_list)
print("NDCG:", total_ndcg)

# NDCG: 0.7834922517044213
