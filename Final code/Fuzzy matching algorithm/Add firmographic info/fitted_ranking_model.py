# import package and module
from ranking_model_features import *

# read data
# you need to change file path
xp = pd.read_parquet("leftdata_old_nodel.parquet")
bq = pd.read_parquet("rightdata_old_nodel.parquet")

xp = xp[xp.company_name_join.apply(lambda x: len(x)>=3)] ## Have to set this up for TFIDF similarity, not lose much data
bq = bq[bq.company_name_join.apply(lambda x: len(x)>=3)] ## Have to set this up for TFIDF similarity

leftdata0 = xp.copy()
rightdata0 = bq.copy()
left=leftdata0
right=rightdata0
n1 = left.shape[0]
n2 = right.shape[0]

# create features and label
feature_1=feature_1(leftdata0,rightdata0)
feature_3=feature_3(leftdata0,rightdata0)
feature_5=feature_5(leftdata0,rightdata0)
feature_6=feature_6(leftdata0,rightdata0)
feature_7=feature_7(leftdata0,rightdata0)
feature_8=feature_8(leftdata0,rightdata0)
feature_9=feature_9(leftdata0,rightdata0)
feature_10=feature_10(leftdata0,rightdata0)
feature_12=feature_12(leftdata0,rightdata0)
query_id=query_id(leftdata0,rightdata0)
label=label(leftdata0,rightdata0)

df=pd.concat([feature_1,feature_3,feature_5,feature_6,feature_7,feature_8,feature_9,feature_10,feature_12,query_id,label],axis=1)

# build ranking model
k =int(leftdata0.shape[0]*0.8)*(rightdata0.shape[0])
train_df = df[:k]  # first 80%
validation_df = df[k:]  # remaining 20%

qids_train = train_df.groupby("query_id")["query_id"].count().to_numpy()
X_train = train_df.drop(["query_id", "label"], axis=1)
y_train = train_df["label"]

qids_validation = validation_df.groupby("query_id")["query_id"].count().to_numpy()
X_validation = validation_df.drop(["query_id", "label"], axis=1)
y_validation = validation_df["label"]

import lightgbm
modelA = lightgbm.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=100)

modelA.fit(
    X=X_train,
    y=y_train,
    group=qids_train,
    eval_set=[(X_validation, y_validation)],
    eval_group=[qids_validation],
    eval_at=5,
    verbose=10,)

# [10]	valid_0's ndcg@5: 0.83462
# [20]	valid_0's ndcg@5: 0.848306
# [30]	valid_0's ndcg@5: 0.871384
# [40]	valid_0's ndcg@5: 0.876812
# [50]	valid_0's ndcg@5: 0.888686
# [60]	valid_0's ndcg@5: 0.892537

lightgbm.plot_importance(modelA)

























