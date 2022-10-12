import numpy as np
import pandas as pd
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


xp = pd.read_parquet("bqDev_1_d030ed26b9-bq.parquet")
bq = pd.read_parquet("bqDev_1_d030ed26b9-xp.parquet")

b=Preprocessing2(bq,xp)
bq, xp=b.data_processing()

xp = xp[xp.company_name_join.apply(lambda x: len(x)>=3)] ## Have to set this up for TFIDF similarity, not lose much data
bq = bq[bq.company_name_join.apply(lambda x: len(x)>=3)] ## Have to set this up for TFIDF similarity

bq = bq.sample(n=10000,random_state=1)
xp = xp[xp.domain_name.isin(bq.domain_name)]

bq.reset_index(drop=True,inplace=True)
xp.reset_index(drop=True,inplace=True)

leftdata0 = xp.copy()
rightdata0 = bq.copy()

print(leftdata0.shape)
print(rightdata0.shape)


def feature_create_formodel(left,right):
  n1 = left.shape[0]
  n2 = right.shape[0]
  
  #feature1
  query1 = list(left["company_name_clean"])
  docs1 = list(right["company_name_clean"])
  # Load the model
  model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L6-cos-v5')
  # Encode query and documents
  query_emb1 = model.encode(query1)
  doc_emb1 = model.encode(docs1)

  # Compute cos similarity between query and all document embeddings
  feature_1 = []
  for i in range(n1):
      scores = util.cos_sim(query_emb1[i], doc_emb1)[0].cpu().tolist()
      for j in range(n2):
          feature_1.append(scores[j])



  # feature 3(if first word in company name matches) 

  left['first_left_name'] = left.company_name_clean.apply(lambda x: x.split()[0] if len(x.split())!=0 else x )#note!!!!
  right['first_right_name'] = right.company_name_clean.apply(lambda x: x.split()[0] if len(x.split())!=0 else x )

  #left['first_left_name'] = left.company_name_clean.apply(lambda x: x.split()[0] )#note!!!!
  #right['first_right_name'] = right.company_name_clean.apply(lambda x: x.split()[0] )


  feature_3=[]

  for i in range(n1):
      for j in range(n2):
          if left.first_left_name[i]==right.first_right_name[j]:
              feature_3.append(1)
          else:
              feature_3.append(0)



  # feature 5(tfidf similarity)

  
  def ngrams(string, n=2):
      ngram = zip(*[string[i:] for i in range(n)])
      return [''.join(ngram) for ngram in ngram]

  #TF-IDF and Vectorization
  vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
  X = vectorizer.fit_transform(left.company_name_join.values)
  Y = vectorizer.transform(right.company_name_join.values)
  feature_5 = []
  for i in range(n1):
      scores = util.cos_sim(X.toarray()[i], Y.toarray())[0].cpu().tolist()
      for j in range(n2):
          feature_5.append(scores[j])


  # feature 6(Sic2_Desc_Similarity)
  nan_index = right[right.sic2_desc.isna()].index
  for i in nan_index:
      right.sic2_desc[i]=right.company_name_clean[i]
  def preprocess(text):
      text = re.sub(r'[^\w\s]', '', text)
      return text
  left['sic2_desc_new'] = left['sic2_desc'].apply(lambda x: preprocess(x))
  right['sic2_desc_new'] = right['sic2_desc'].apply(lambda x: preprocess(x))
  query3 = list(left["sic2_desc_new"])
  docs3 = list(right["sic2_desc_new"])
  model3 = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L6-cos-v5')
  query_emb3 = model3.encode(query3)
  doc_emb3 = model3.encode(docs3)
  # Compute cos similarity between query and all document embeddings
  feature_6 = []
  for i in range(n1):
      scores = util.cos_sim(query_emb3[i], doc_emb3)[0].cpu().tolist()
      for j in range(n2):
          feature_6.append(scores[j])
 
  # feature 7(if # 0f company_name match)
  left["num_of_company_name"] = left["company_name_clean"].apply(lambda x:len(x.split())) 
  right["num_of_company_name"] = right["company_name_clean"].apply(lambda x:len(x.split())) 
  feature_7=[]
  for i in range(n1):
      for j in range(n2):
          if left.num_of_company_name[i]==right.num_of_company_name[j]:
              feature_7.append(1)
          else:
              feature_7.append(0)
  
  # feature 8&9(# 0f company_name)
  feature_8=[]
  for i in range(n1):
      L = [left.num_of_company_name[i]]*n2
      feature_8.extend(L)


  feature_9=[]
  for i in range(n1):
      feature_9.extend(right.num_of_company_name.values)

 



  feature_affinegap_companyname=[]
  for i in range(len(left)):

          
      params = dict(fuzzy_left_on=['company_name_join'], fuzzy_right_on=['company_name_join'],
              #exact_left_on=['state', 'city'], exact_right_on=['state', 'city'],
              fun_diff=[affinegap.normalizedAffineGapDistance], is_keep_debug=True)  
      j = top5.MergeTop1(left.iloc[i:i+1],right, **params)
      resulttop5_AffineGapDistance = j.merge()

      joindata=resulttop5_AffineGapDistance['merged']#.dropna()
      joindata = joindata.reset_index() # index is not continuous
      joindata.drop(columns=['index'],inplace=True)
      
      feature_affinegap_companyname=feature_affinegap_companyname+joindata["__top1diff__company_name_join"].tolist()
 # feature_12
  left['address']=left.apply(lambda x:str(x['state'])+' '+str(x['city']),axis=1)
  right['address']=right.apply(lambda x:str(x['state'])+' '+str(x['city']),axis=1)
  feature_12=[]
  for i in range(n1):
      for j in range(n2):
          if left.address[i]==right.address[j]:
              feature_12.append(1)
          else:
              feature_12.append(0)

  #query id
  query_id=[]
  a=0
  for i in range(n1):
      for j in range(n2):
          query_id.append(a)
      a+=1

#label  
  label=[]
  for i in range(n1):
      for j in range(n2):
          if left.domain_name[i]==right.domain_name[j]:
              label.append(1)
          if left.domain_name[i]!=right.domain_name[j]:
              label.append(0)


  df = pd.DataFrame({
    "query_id":query_id,
    "companyname_Bert_similarity":feature_1,
  #  "address_Bert_similarity":feature_2,
   "if_firstword_match":feature_3,
    "companyname_tfidf_similarity":feature_5,
    "Sic_Bert_similarity":feature_6,
    "if_#word_companyname_match":feature_7,
    "left_#word_companyname":feature_8,
    "right_#word_companyname":feature_9,
     "companyname_affinegap":feature_affinegap_companyname,
   # "address_affinegap":feature_affinegap_address,
    "if_address_match":feature_12,

    "relevance":label
  })
  return df

df=feature_create_formodel(leftdata0,rightdata0)

k =int(leftdata0.shape[0]*0.8)*(rightdata0.shape[0])
train_df = df[:k]  # first 80%
validation_df = df[k:]  # remaining 20%

qids_train = train_df.groupby("query_id")["query_id"].count().to_numpy()
X_train = train_df.drop(["query_id", "relevance"], axis=1)
y_train = train_df["relevance"]

qids_validation = validation_df.groupby("query_id")["query_id"].count().to_numpy()
X_validation = validation_df.drop(["query_id", "relevance"], axis=1)
y_validation = validation_df["relevance"]

import xgboost as xgb

model = xgb.sklearn.XGBRanker(
        objective='rank:ndcg',
        learning_rate=0.1,      
        n_estimators=100)

model.fit(
    X=X_train,
    y=y_train,
    group=qids_train,
    eval_set=[(X_validation, y_validation)],
    eval_group=[qids_validation],
    eval_metric = "ndcg@5",
    #eval_at=5,
    verbose=10)

# NDCG: 0.9
