import pandas as pd
import re, string
from nltk.tokenize import word_tokenize

# individual function
def preprocess(text):
    text = text.lower()
    text = text.strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_whitespace(text):
        return text.replace(" ", "")

def generate_preprocessed_data(bq):
        bq = bq.reset_index() # check if index is not continuous
        bq.drop(columns=['index'],inplace=True)
        bq["company_name_clean"] = bq["company_name"].apply(lambda x: del_some(x))
        bq[ "company_name_join"] = bq["company_name_clean"].apply(lambda x: remove_whitespace(x))
        bq.loc[bq["company_name_clean"]=="","company_name_clean"]=bq["company_name"]
        bq.loc[bq["company_name_join"]=="","company_name_join"]=bq["company_name"]
        return bq

def generate_preprocessed_data1(bq):
    bq = bq.reset_index() # check if index is not continuous
    bq.drop(columns=['index'],inplace=True)
    bq["company_name_clean"] = bq["company_name"].apply(lambda x: preprocess(x))
    bq[ "company_name_join"] = bq["company_name_clean"].apply(lambda x: remove_whitespace(x))
    bq.loc[bq["company_name_clean"]=="","company_name_clean"]=bq["company_name"]
    bq.loc[bq["company_name_join"]=="","company_name_join"]=bq["company_name"]




    return bq

def preprocess1(text):
        text = text.title()
        text = re.sub(r'[^\w\s]', '', text)
        return text

def industry(x):
        return x.split(';')[0]

def del_some(string):
        word_t = word_tokenize(preprocess(string))
        if len(word_t) != 0:
            if word_t[-1] == 'inc':
                word_t.pop()
            elif word_t[-1] == 'llc':
                word_t.pop()
            elif word_t[-1] == 'lnc':
                word_t.pop()

        return " ".join(word_t)

def print_shape(dataset):
    print(dataset)

def check(dataset):
    print(dataset.loc[dataset["company_name_join"] == ""])
    print(dataset.loc[dataset["company_name_clean"] == ""])


# preprocessing


# this one is for firmographical
class Preprocessing1:

    def __init__(self,expanded_bq,dfj_bq,dfj_xp):
        self.expanded_bq = expanded_bq
        self.dfj_bq = dfj_bq
        self.dfj_xp = dfj_xp

    def data_processing(self):
        expanded_bq_0 = self.expanded_bq.iloc[:, [3, 8, 22, 27, 28, 66, 67, 68, 69, 70, 71]]
        expanded_bq_1 = expanded_bq_0.rename(columns={'bq_website': 'domain_name',
                                                      'bq_company_name': 'company_name',
                                                      'bq_company_address1_state': 'state',
                                                      'bq_company_address1_city': 'city',
                                                      'bq_sic_sector_name': 'sic2_desc',
                                                      'bq_net_income_mr': 'bq_operating_income_mr',
                                                      'bq_net_income_growth_1yr_mr': 'bq_operating_income_growth_1yr_mr'})
        expanded_bq_2 = generate_preprocessed_data(expanded_bq_1)
        expanded_bq_2['company_name'] = expanded_bq_2['company_name'].apply(lambda x: preprocess1(x))
        bq = generate_preprocessed_data(self.dfj_bq)
        bq['company_name'] = bq['company_name'].apply(lambda x: preprocess1(x))
        xp = generate_preprocessed_data(self.dfj_xp)
        xp['company_name'] = xp['company_name'].apply(lambda x: preprocess1(x))
        return expanded_bq_2,bq,xp


# this one is for ndcg of models but not ranking model
class Preprocessing2:

    def __init__(self,dfj_bq,dfj_xp):
        self.dfj_bq = dfj_bq
        self.dfj_xp = dfj_xp

    def data_processing(self):

        bq = generate_preprocessed_data(self.dfj_bq)
        bq['company_name'] = bq['company_name'].apply(lambda x: preprocess1(x))
        xp = generate_preprocessed_data(self.dfj_xp)
        xp['company_name'] = xp['company_name'].apply(lambda x: preprocess1(x))
        rightdata = bq[bq.domain_name.notnull()]
        leftdata = xp.loc[xp.domain_name.isin(rightdata.domain_name)]
        return rightdata,leftdata


# this one is for ndcg of ranking model
class Preprocessing3:

    def __init__(self,dfj_bq,dfj_xp):
        self.dfj_bq = dfj_bq
        self.dfj_xp = dfj_xp

    def data_processing(self):

        bq = generate_preprocessed_data1(self.dfj_bq)
        bq['company_name'] = bq['company_name'].apply(lambda x: preprocess1(x))
        xp = generate_preprocessed_data1(self.dfj_xp)
        xp['company_name'] = xp['company_name'].apply(lambda x: preprocess1(x))
        rightdata = bq[bq.domain_name.notnull()]
        leftdata = xp.loc[xp.domain_name.isin(rightdata.domain_name)]
        return rightdata,leftdata





