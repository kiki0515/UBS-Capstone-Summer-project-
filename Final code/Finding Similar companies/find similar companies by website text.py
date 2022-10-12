import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics import f1_score

#nltk.download('stopwords')
#nltk.download('wordnet')


pd.set_option('display.max_columns', None)  # display all columns
desired_width = 320
pd.set_option('display.width', desired_width)

# Download Data
WebsiteTxt1 = pd.read_parquet('E:/capstone/firmographic/WebsiteTxt_0_True_True_fintent_mongo_cb_202203_c9e93402ee-search.parquet')
WebsiteTxt1.dropna(subset=['website_summary'],inplace=True)
WebsiteTxt1 = WebsiteTxt1.reset_index()
WebsiteTxt1.drop(columns=['index'],inplace=True)


# Preprocess/Clean
cachedStopWords = stopwords.words('english')
toke = RegexpTokenizer(r'\w+')
lemma = WordNetLemmatizer()
def preprocess(text, cachedStopWords=cachedStopWords, toke=toke, lemma=lemma):
    # Remove numbers # Later tooknizer show that there are many numbers
    text0 = re.sub(r'\d+', '', text)
    # Tokenization while ignoring punctuations
    text1 = toke.tokenize(text0)
    # Lemmatisation and Lower casing
    text2 = [lemma.lemmatize(word.lower(), pos='v') for word in text1]
    # Removing Stop words
    text3 = [word for word in text2 if word not in cachedStopWords]
    # untokenized to string
    text4 = TreebankWordDetokenizer().detokenize(text3)
    return text4
WebsiteTxt1['website_summary'] = WebsiteTxt1.content_search.apply(lambda x: preprocess(x))
WebsiteTxt1.drop([9356],inplace=True)


WebsiteTxt1 = WebsiteTxt1.reset_index()
WebsiteTxt1.drop(columns=['index'],inplace=True)
WebsiteTxt2 = WebsiteTxt1


#TF-IDF and Vectorization
vec = TfidfVectorizer()
X = vec.fit_transform(WebsiteTxt1['content_search'])
Y = vec.transform(WebsiteTxt2.content_search.values)

# calculate the similarity between two vectors of TF-IDF values the Cosine Similarity is usually used.
# result matrix in a very sparse terms and Scikit-learn deals with this nicely by returning a sparse CSR matrix.
# Run the optimized cosine similarity function.
t1 = time.time()
matches = awesome_cossim_topn(X, Y.transpose(), ntop=3, lower_bound=0.0)
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

    left_content_search_txt = np.empty([nr_matches], dtype=object)
    left_domain = np.empty([nr_matches], dtype=object)
    right_content_search_txt = np.empty([nr_matches], dtype=object)
    right_domain = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)

    for index in range(0, nr_matches):
        left_content_search_txt[index] = A['content_search'].values[sparserows[index]]
        left_domain[index] = A['domain_name'].values[sparserows[index]]
        right_content_search_txt[index] = B['content_search'].values[sparsecols[index]]
        right_domain[index] = B['domain_name'].values[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]

    return pd.DataFrame({'left_content_search_txt': left_content_search_txt,
                         'right_content_search_txt': right_content_search_txt,
                         'left_domain': left_domain,
                         'right_domain':right_domain,
                         'similarity': similairity})

# store the matches into new dataframe called matched_df
matches_df = get_matches_df(matches, WebsiteTxt1, WebsiteTxt2, top=0)

# check ntop=3 and there is not less than 3
result1 = np.reshape(matches.nonzero()[0],(int(matches_df.shape[0]/3),3))
s = 0
for i in range(len(result1)):
    s += (result1[i][0]!=result1[i][2])
assert s==0

result2 = np.reshape(matches.nonzero()[1],(int(matches_df.shape[0]/3),3))
s2 = 0
L2 = [] # Store left-side invalid observations
L = [] # Store right-side invalid observations
for i in range(len(result2)-1):
    s2 += (result2[i][0]!=(result2[i+1][0]-1))
    if result2[i][0]!=(result2[i+1][0]-1) and result2[i+1][0]!=(result2[i+2][0]-1):
        L.append(i+1)
        L2.append((i+1)*3)
        L2.append((i+1)*3+1)
        L2.append((i+1)*3+2)
assert s2==0
print(len(L))
print(len(L2))
print(WebsiteTxt1.iloc[L,:])
print(matches_df.iloc[L2,:])

# Output most 2 similar excel
matches_df2 = matches_df.copy()
matches_df2.drop(L2,inplace=True)
data = pd.DataFrame(np.reshape(matches_df2.right_domain.values,(int(matches_df2.shape[0]/3),3)))
data.columns = ["Domain Name", "Most Similar","Second Similar"]
data.to_excel('C:/Users/Kaiyun Kang/Downloads/similar_company_output2.xlsx')

# save WebsiteTxt_new: delete invalid observation on Content_search
WebsiteTxt_new = WebsiteTxt1.drop(L)
WebsiteTxt_new = WebsiteTxt_new.reset_index()
WebsiteTxt_new.drop(columns=['Unnamed: 0'],inplace=True)
WebsiteTxt_new.to_parquet('E:/capstone/firmographic/WebsiteTxt_new_17614.parquet')


