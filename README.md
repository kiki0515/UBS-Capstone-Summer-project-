# UBS Capstone Summer project - Predictive modeling team


### There are three tasks we've done.


# 1. Fuzzy matching algorithm
Folders: 
- Ndcg calculation: NDCG@5 scores of three different types of models: Databolt smart join algorithm, NLP embedding methods(TF-IDF, Word2Vec, SBert), and Ranking models(XGB Ranker, LGBM Ranker).
- Add firmographic info: includes codes for LGBM Ranker model and merge datasets from different vendors into one comprehensive dataset. 

Source code:
- add_firmographic_info_ranking_model.py: main code to create complete merged data
- preprocessing.py: code for data preprocessing
- ranking_model_features.py: code for feature engineering in the LGBM ranking model. 


# 2. Finding similar companies by website text
Source code: 
- find similar companies by website text.py: code for TF-IDF embedding technique to find most similar two companies.

# 3. CapRaise model
Folders:
- Input: Folder containing all input files. This includes the original data.
- Output: Folder containing all output files. This includes preprocessed data and results such as cross-validated scores.

Source code:
- Capraise_Data_Preprocessing.py: The script for data preprocessing
- Capraise_Model_AutoML.ipynb: Jupyter Notebook of the Auto Sklearn model. CAUTION: This version can be run on Google Colab only.
- Capraise_Model_Baseline_PiML.ipynb: Jupyter Notebook of the Baseline Model and PiML models. It needs the piml package to run.


