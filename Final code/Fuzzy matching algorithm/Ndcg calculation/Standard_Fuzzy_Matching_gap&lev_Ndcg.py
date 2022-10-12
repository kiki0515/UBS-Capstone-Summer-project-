
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
dfj_bq= pd.read_parquet("bqDev_1_d030ed26b9-bq.parquet")
dfj_xp= pd.read_parquet("bqDev_1_d030ed26b9-xp.parquet")
b= Preprocessing2(dfj_bq,dfj_xp)
rightdata,leftdata=b.data_processing()

#AffineGapDistance
ndcg_score_avg = []
for i in range(len(leftdata)):
    params = dict(fuzzy_left_on=['company_name_join'], fuzzy_right_on=['company_name_join'],

                  fun_diff=[affinegap.normalizedAffineGapDistance],
                  is_keep_debug=True)
    j = topn.MergeTop1(leftdata.iloc[i:i + 1], rightdata, **params)
    resulttop5_AffineGapDistance = j.merge()

    joindata = resulttop5_AffineGapDistance['merged']
    joindata = joindata.reset_index()
    joindata.drop(columns=['index'], inplace=True)

    joindata.loc[joindata['domain_name_right'] == joindata['domain_name'], 'Match or not'] = 1
    joindata.loc[joindata['domain_name_right'] != joindata['domain_name'], 'Match or not'] = 0

    joindata["y_score"] = 100 - joindata["__top1diff__company_name_join"]

    ndcg_score_avg.append(ndcg_score([joindata["Match or not"]], [joindata["y_score"]], k=5))


from statistics import mean
print("average ndcg_score for AffineGapDistance :",mean(ndcg_score_avg))

#average ndcg_score for AffineGapDistance : 0.7400615519138096

#jellyfish


ndcg_score_avg_levenshtein_distance = []
for i in range(len(leftdata)):
    params = dict(fuzzy_left_on=['company_name_join'], fuzzy_right_on=['company_name_join'],

                  fun_diff=[jellyfish.levenshtein_distance], is_keep_debug=True)

    j1 = topn.MergeTop1(leftdata.iloc[i:i + 1], rightdata, **params)
    resulttop5_levenshtein_distance = j1.merge()

    joindata = resulttop5_levenshtein_distance['merged']
    joindata = joindata.reset_index()  
    joindata.drop(columns=['index'], inplace=True)

    joindata.loc[joindata['domain_name_right'] == joindata['domain_name'], 'Match or not'] = 1
    joindata.loc[joindata['domain_name_right'] != joindata['domain_name'], 'Match or not'] = 0

    joindata["y_score"] = 100 - joindata["__top1diff__company_name_join"]
    ndcg_score_avg_levenshtein_distance.append(ndcg_score([joindata["Match or not"]], [joindata["y_score"]], k=5))



print("average ndcg_score for levenshtein_distance :",mean(ndcg_score_avg_levenshtein_distance))
#average ndcg_score for levenshtein_distance : 0.6535073820129174
