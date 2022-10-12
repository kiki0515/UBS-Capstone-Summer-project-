import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn
from sklearn.metrics import ndcg_score
import math

pd.set_option('display.max_columns', None)  # display all columns
desired_width = 320
pd.set_option('display.width', desired_width)


left = pd.read_parquet('E:/capstone/firmographic/leftdata_old4.parquet')
left = left.reset_index()
left.drop(columns=['index'],inplace=True)

right = pd.read_parquet('E:/capstone/firmographic/rightdata_old4.parquet')
right = right.reset_index()
right.drop(columns=['index'],inplace=True)

print(left.shape) #(7926, 7)
print(right.shape) #(226514, 7)


df1 = left[left.company_name_join.apply(lambda x: len(x)>=3)]
training_set_bq = right[right.company_name_join.apply(lambda x: len(x)>=3)]
print(df1.shape)

#ngrams
def ngrams(string, n=2):
    ngram = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngram]

#TF-IDF and Vectorization
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
X = vectorizer.fit_transform(df1.company_name_join.values)
Y = vectorizer.transform(training_set_bq.company_name_join.values)

t1 = time.time()
matches = awesome_cossim_topn(X, Y.transpose(), ntop=5, lower_bound=0.0)
t = time.time()-t1
print("SELFTIMED:", t)


def get_matches_df(sparse_matrix, A, B, top=100):
    non_zeros = sparse_matrix.nonzero()
    sparserows = non_zeros[0] #index of A
    sparsecols = non_zeros[1] #corresponding index of B

    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size

    left_company_name = np.empty([nr_matches], dtype=object)
    left_domain = np.empty([nr_matches], dtype=object)
    left_company = np.empty([nr_matches], dtype=object)
    right_company_name = np.empty([nr_matches], dtype=object)
    right_domain = np.empty([nr_matches], dtype=object)
    right_company = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)

    for index in range(0, nr_matches):
        left_company_name[index] = A['company_name_join'].values[sparserows[index]]
        left_domain[index] = A['domain_name'].values[sparserows[index]]
        left_company[index] = A['company_name'].values[sparserows[index]]
        right_company_name[index] = B['company_name_join'].values[sparsecols[index]]
        right_domain[index] = B['domain_name'].values[sparsecols[index]]
        right_company[index] = B['company_name'].values[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]

    return pd.DataFrame({'left_company_name_join': left_company_name,
                         'right_company_name_join': right_company_name,
                         'left_company_name':left_company,
                         'right_company_name':right_company,
                         'left_domain': left_domain,
                         'right_domain':right_domain,
                         'similarity': similairity})

# store the matches into new dataframe called matched_df and printing 10 samples
matches_df_5 = get_matches_df(matches, df1,training_set_bq, top=0)
print(matches_df_5)

# check
result1 = np.reshape(matches.nonzero()[0],(int(matches_df_5.shape[0]/5),5))
s = 0
for i in range(len(result1)):
    s += (result1[i][0]!=result1[i][4])
assert s==0


matches_df_5["actual"] = 0
matches_df_5.actual = matches_df_5.actual.where(matches_df_5.left_domain != matches_df_5.right_domain,1)
matches_df_5["predict_rank"] = [1,2,3,4,5]*int(matches_df_5.shape[0]/5)

# DCG
matches_df_5["reli/log2(i+1)"] = np.nan
for i in range(matches_df_5.shape[0]):
    matches_df_5.iloc[i,-1] = matches_df_5.actual[i]/math.log2(matches_df_5.predict_rank[i]+1)

print(matches_df_5)
print(matches_df_5["actual"].value_counts())
print(matches_df_5["reli/log2(i+1)"].value_counts())



#IDCG
List1 = []
for j in range(df1.shape[0]):
    List2 = sorted((df1.iloc[j,0]==training_set_bq.iloc[:,0]).values.astype(int),reverse=True)[:5]
    List1.extend(List2)
matches_df_5["ideal_rank"] = List1

matches_df_5["rel_ideal/log2(i+1)"] = np.nan
for i in range(matches_df_5.shape[0]):
    matches_df_5.iloc[i,-1] = matches_df_5.ideal_rank[i]/math.log2(matches_df_5.predict_rank[i]+1)

NDCG = 0.0
for i in range(0,matches_df_5.shape[0],5):
    DCG = sum(matches_df_5.iloc[i:i+5,-3])
    IDCG = sum(matches_df_5.iloc[i:i+5,-1])
    if IDCG ==0.0:
        NDCG += 0.0
    else:
        NDCG += (DCG / IDCG)

NDCG_result = NDCG/df1.shape[0]

print(NDCG_result)


