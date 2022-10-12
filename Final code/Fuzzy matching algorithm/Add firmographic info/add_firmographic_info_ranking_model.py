# import package and module
from preprocessing import *
from ranking_model_features import *
from fitted_ranking_model import *

# read data
# you need to change path
Website_Txt= pd.read_parquet("./Data/WebsiteTxt_0_True_True_fintent_mongo_cb_202203_c9e93402ee-search.parquet")
xp= pd.read_parquet("./Data/bqDev_0_3040c55535-xp.parquet")
bq= pd.read_parquet("./Data/bqDev_0_3040c55535-bq.parquet")
financials= pd.read_parquet("./Data/bq_0_3040c55535-financials.parquet")
expanded_bq = pd.read_csv("./Data/fintent.csv")

# preprocess data
Preprocess_data = Preprocessing1(expanded_bq, bq, xp)
expanded_bq, bq, xp = Preprocess_data.data_processing()

# combine new bq dataset and old bq dataset
old_bq = pd.merge(bq,financials,how='inner',on='bq_id')
old_bq_0 = old_bq.iloc[:,[1,2,3,4,5,6,8,9,10,11,12,13]]
bq_dataset = pd.concat([expanded_bq,old_bq_0])

# Split bq into bq_valid and bq_null
# bq_valid - the part with non-null value in domain_name
# bq_null - the part with empty in domain_name
bq_valid = bq_dataset[bq_dataset['domain_name'].notna()]
bq_null = bq_dataset[bq_dataset['domain_name'].isna()]

# domain exact join bq_valid –> True
Website_Txt_0 = pd.merge(Website_Txt,bq_valid,how='inner',on='domain_name')
With_Firmographics_0 = Website_Txt_0.drop_duplicates(subset=['domain_name'])
With_Firmographics_Guaranteed_Correct = With_Firmographics_0[['domain_name',
                                                             'company_name',
                                                             'state',
                                                             'city',
                                                             'sic2_desc',
                                                             'bq_current_employees_plan_mr',
                                                             'bq_current_employees_plan_growth_1yr_mr',
                                                             'bq_revenue_mr',
                                                             'bq_revenue_growth_1yr_mr',
                                                             'bq_operating_income_mr',
                                                             'bq_operating_income_growth_1yr_mr']]

# domain exact join bq_valid –> False
Website_Txt_1 = Website_Txt[~Website_Txt['domain_name'].isin(With_Firmographics_Guaranteed_Correct['domain_name'])]

# domain exact join xp –> True
Website_Txt_2 = pd.merge(Website_Txt_1,xp,how='inner',on='domain_name')
Website_Txt_2_No_Duplicate = Website_Txt_2.drop_duplicates(subset=['domain_name'])

# comp name fuzzy join bq_null(ranking model)
Website_Txt_2_No_Duplicate = Website_Txt_2_No_Duplicate[Website_Txt_2_No_Duplicate.company_name_join.apply(lambda x: len(x)>=3)] ## Have to set this up for TFIDF similarity, not lose much data
bq_null = bq_null[bq_null.company_name_join.apply(lambda x: len(x)>=3)]
bq_null.reset_index(drop=True,inplace=True)
Website_Txt_2_No_Duplicate.reset_index(drop=True,inplace=True)

left= Website_Txt_2_No_Duplicate
right= bq_null

df = pd.DataFrame()
for i in range(len(left)):
 featurei=feature_create(left.iloc[i:i+1].reset_index(drop=True),right)
 a=modelA.predict(featurei)
 newright=right.copy()
 newright["predicted_ranking"] = a
 dfi=pd.concat([left.iloc[i:i+1].reset_index(drop=True), newright.sort_values("predicted_ranking", ascending=False).reset_index(drop=True).iloc[0:1]], axis=1)
 df=df.append(dfi, ignore_index=True)

With_Firmographics_Not_Guaranteed_0 = df[['domain_name',
                                          'company_name',
                                          'state',
                                          'city',
                                          'sic2_desc',
                                          'bq_current_employees_plan_mr',
                                          'bq_current_employees_plan_growth_1yr_mr',
                                          'bq_revenue_mr',
                                          'bq_revenue_growth_1yr_mr',
                                          'bq_operating_income_mr',
                                          'bq_operating_income_growth_1yr_mr']]


# domain exact join xp --> False
Website_Txt_4 = Website_Txt_1[~Website_Txt_1['domain_name'].isin(With_Firmographics_Not_Guaranteed_0['domain_name'])]
No_firm = Website_Txt_4[['domain_name']]

# Combine
Output = pd.concat([With_Firmographics_Guaranteed_Correct, With_Firmographics_Not_Guaranteed_0, No_firm])
# Output.to_csv('Output_partial_ranking_model.csv')
# Output.to_parquet('Output_partial_ranking_model.parquet')