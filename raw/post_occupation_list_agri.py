# Import necessary modules
import numpy as np
import pandas as pd
import nltk
import csv
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer 
import datetime

import os
import re
import sys
import platform
import collections
import shutil
import math
import multiprocessing
import os.path
from gensim import corpora, models
from gensim.models import Word2Vec, keyedvectors 
from gensim.models.word2vec import LineSentence
from sklearn.metrics.pairwise import cosine_similarity

data_folder = "C:\\Users\\norih\\Desktop\\workingdata_bdjob\\"


# import job post data
file_open = data_folder + "Agro - Jobs.xlsx"
data = pd.read_excel(file_open)
data.rename(columns={'Job Responsibility': 'job_desc', 'Job Title': 'title'}, inplace=True)
data.rename(columns={'jdes': 'job_desc', 'jtitle': 'title'}, inplace=True)
data["job_desc"].fillna(" ", inplace = True)
data["title"].fillna(" ", inplace = True)

# tokenized list of job descriptions and job titles
w_tokenizer = RegexpTokenizer(r'\w+')
stop_words = stopwords.words('english')
tokenized_job_desc = [[w for w in w_tokenizer.tokenize(data['job_desc'][i].lower()) if not w in stop_words] for i in range(len(data))]
tokenized_job_title = [[w for w in w_tokenizer.tokenize(data['title'][i].lower()) if not w in stop_words] for i in range(len(data))]

# construct CBOW model from job descriptions and job titles
dim_model = 300
model = Word2Vec(tokenized_job_desc + tokenized_job_title,
                 size=dim_model, 
                 window=5, 
                 min_count=5, 
                 workers=multiprocessing.cpu_count())

model.init_sims(replace=True)
word_all = model.wv # set of all words in the model

# Occupation (ISCO-08) ---------------

data_occupation = data[:]

# From the list of ISCO-08, I drop the following since they are too formal for online job market:
# - firefighters, police officers, prison guards, security guards
# - armed forces officers
# - customs and border inspectors, government tax and excise officials, government social benefits officials,
#   government licensing officials, police inspectors and detectives
# - legislators, senior government officials, traditional chiefs and heads of villages, senior officials of special-interest organizations

isco08_data = pd.read_excel(data_folder + 'ISCO08_code_noformal.xlsx')
# keep isco-08 titles and codes and rename them
isco08_data = isco08_data[['ISCO-08 Title', 'ISCO-08 Code']].rename(index=str, columns={'ISCO-08 Title': 'isco08_title', 'ISCO-08 Code': 'isco08_code'})
# keep rows if there exists isco-08 titles
isco08_data = isco08_data[-isco08_data.isco08_title.isna()]
isco08_data.isco08_title = [w.lower() for w in isco08_data.isco08_title]

# derive the word vectors of occupations in ISCO08
regex = re.compile('[^a-zA-Z0-9]')
isco08_vector_list = []
for occupation in isco08_data.isco08_title:
    occupation = regex.sub(' ', occupation)
    tokens_occupation = [w for w in re.split(' ',occupation) if w in word_all if not w in stop_words] 
    vector_to_match = np.zeros(dim_model)
    for w in tokens_occupation:
        vector_to_match += model[w]
    isco08_vector_list.append(vector_to_match)

# calculate the distances from each job title to word vectors calculated above,
# and assign the most similar occupation to the occupation of the job
occupation_list = []

aaaa = data_occupation.title.str.isnumeric()
bbbb = data_occupation[-aaaa]

count = -1
for title in bbbb.title:
    count = count + 1
    title = regex.sub(' ', title).lower()
    tokens_title = [w for w in re.split(' ',title) if w in word_all if not w in stop_words] 
    # If the job title is the same as the occupation title in ISCO08,
    # simply assign it.
    if title in isco08_data.isco08_title:
        occupation_index = [w == title for w in isco08_data.isco08_title]
        occupation_temp = [isco08_data.isco08_title[occupation_index].item(), isco08_data.isco08_code[occupation_index].item(), np.nan]
        occupation_list.append(occupation_temp)
    else:
        # nan if words in the job title are not in the CBOW word list
        if tokens_title == []:
            occupation_temp = [np.nan, np.nan, np.nan]
            occupation_list.append(occupation_temp)
        # otherwise, we calculate the distances
        else:
            vector_title = np.zeros(dim_model)
            for w in tokens_title:
                vector_title += model[w]
            list_temp = np.inner(isco08_vector_list, vector_title) / ((np.ones(len(isco08_vector_list)) * np.linalg.norm(vector_title)) * np.linalg.norm(isco08_vector_list, axis=1))
            occupation_index = np.nanargmax(list_temp)
            occupation_temp = [isco08_data.isco08_title[occupation_index], isco08_data.isco08_code[occupation_index], list_temp[occupation_index]]
            occupation_list.append(occupation_temp)

# merge the occupation information derived above
occupation_list = pd.DataFrame(occupation_list, columns = ['Occupation', 'ISCO08code', 'cosine_similarity'])
data_occupation['isco08_occupation'] = occupation_list.Occupation
data_occupation['isco08_code'] = occupation_list.ISCO08code
data_occupation['isco08_cosine_similarity'] = occupation_list.cosine_similarity

# assign digit numbers in ISCO08 data to each job based on the classified occupations
isco08_one_digit   = isco08_data[(isco08_data['isco08_code'] / 10 < 1) & (isco08_data['isco08_code'] / 1 >= 1)].rename(index=str, columns={'Title EN': 'isco08_one_digit_occupation'})
isco08_two_digit   = isco08_data[(isco08_data['isco08_code'] / 100 < 1) & (isco08_data['isco08_code'] / 10 >= 1)].rename(index=str, columns={'Title EN': 'isco08_two_digit_occupation'})
isco08_three_digit = isco08_data[(isco08_data['isco08_code'] / 1000 < 1) & (isco08_data['isco08_code'] / 100 >= 1)].rename(index=str, columns={'Title EN': 'isco08_three_digit_occupation'})
isco08_four_digit  = isco08_data[(isco08_data['isco08_code'] / 10000 < 1) & (isco08_data['isco08_code'] / 1000 >= 1)].rename(index=str, columns={'Title EN': 'isco08_four_digit_occupation'})

data_occupation['isco08_one_digit'] = np.floor(data_occupation['isco08_code'] / 1000)
data_occupation['isco08_two_digit'] = np.floor(data_occupation['isco08_code'] / 100)
data_occupation['isco08_three_digit'] = np.floor(data_occupation['isco08_code'] / 10)
data_occupation['isco08_four_digit'] = data_occupation['isco08_code']
data_occupation = data_occupation.merge(isco08_one_digit, left_on = 'isco08_one_digit', right_on = 'isco08_code', how = 'left')
data_occupation = data_occupation.merge(isco08_two_digit, left_on = 'isco08_two_digit', right_on = 'isco08_code', how = 'left')
data_occupation = data_occupation.merge(isco08_three_digit, left_on = 'isco08_three_digit', right_on = 'isco08_code', how = 'left')
data_occupation = data_occupation.merge(isco08_four_digit, left_on = 'isco08_four_digit', right_on = 'isco08_code', how = 'left')

# Occupation (O*NET-SOC) ---------------

# import O*NET data
onet_data = pd.read_csv(data_folder + 'Occupation Data.txt', sep="\t")
onet_data['O*NET-SOC Code'] = [w.replace('-','').replace('.','') for w in onet_data['O*NET-SOC Code']]
onet_job_titles = pd.read_csv(data_folder + 'title2soc.txt', sep="\t", header=None).rename(index=str, columns={0: 'job_title', 1: 'soc_title', 2: 'soc_code'})

# derive the word vectors of occupations in O*NET
onet_vector_list = []
for occupation in onet_job_titles.job_title:
    occupation = regex.sub(' ', occupation)
    tokens_occupation = [w for w in re.split(' ',occupation) if w in word_all if not w in stop_words] 
    vector_to_match = np.zeros(dim_model)
    for w in tokens_occupation:
        vector_to_match += model[w]
    onet_vector_list.append(vector_to_match)

# calculate the distances from each job title to word vectors calculated above,
# and assign the most similar occupation to the occupation of the job
occupation_list = []
for title in data_occupation.title:
    title = regex.sub(' ', title).lower()
    tokens_title = [w for w in re.split(' ',title) if w in word_all if not w in stop_words] 
    # If the job title is the same as the occupation title in O*NET,
    # simply assign it.
    if title in list(onet_job_titles.job_title):
        occupation_index = [w == title for w in onet_job_titles.job_title]
        occupation_temp = [onet_job_titles.soc_title[occupation_index].item(), onet_job_titles.soc_code[occupation_index].item(), np.nan]
        occupation_list.append(occupation_temp)
    else:
        # nan if words in the job title are not in the CBOW word list
        if tokens_title == []:
            occupation_temp = [np.nan, np.nan, np.nan]
            occupation_list.append(occupation_temp)
        # otherwise, we calculate the distances
        else:
            vector_title = np.zeros(dim_model)
            for w in tokens_title:
                vector_title += model[w]
            list_temp = np.inner(onet_vector_list, vector_title) / ((np.ones(len(onet_vector_list)) * np.linalg.norm(vector_title)) * np.linalg.norm(onet_vector_list, axis=1))
            occupation_index = np.nanargmax(list_temp)
            occupation_temp = [onet_job_titles.soc_title[occupation_index], onet_job_titles.soc_code[occupation_index], list_temp[occupation_index]]
            occupation_list.append(occupation_temp)

# merge the occupation information derived above
occupation_list = pd.DataFrame(occupation_list, columns = ['Occupation', 'SOC_code', 'cosine_similarity'])
data_occupation['onet_occupation_sub'] = occupation_list.Occupation
data_occupation['onet_code'] = occupation_list.SOC_code
data_occupation['onet_cosine_similarity'] = occupation_list.cosine_similarity

# clean dataframe for output
# drop if job id is strange
weird_jid = [type(w) != str for w in data_occupation.jid]
data_occupation_output = data_occupation[weird_jid]
data_occupation_output.job_desc = [w.replace('\r\n','') for w in data_occupation_output.job_desc]
data_occupation_output = data_occupation_output.merge(onet_data.drop('Description', axis = 1).rename(index=str, columns={'Title': 'onet_occupation'}), left_on = 'onet_code', right_on = 'O*NET-SOC Code')
data_occupation_output['onet_four_digit'] = [w.replace('-','')[0:4] for w in data_occupation_output.onet_code]
data_occupation_output['onet_six_digit'] = [w.replace('-','')[0:5] + '0' for w in data_occupation_output.onet_code]

# assign SOC code (4-digit and 6-digit) to each job
soc_code = pd.read_excel(data_folder + 'soc_structure_2010.xls')
soc_code_4digit = soc_code.dropna(subset=['Minor Group']).drop(['Major Group', 'Broad Group', 'Detailed Occupation'], axis = 1).rename(index=str, columns={'Minor Group': 'four_digit', 'soc_occupation': 'soc_occupation_4digit'})
soc_code_4digit.four_digit = [w.replace('-','')[0:4] for w in soc_code_4digit.four_digit]

soc_code_6digit = soc_code.dropna(subset=['Broad Group']).drop(['Major Group', 'Minor Group', 'Detailed Occupation'], axis = 1).rename(index=str, columns={'Broad Group': 'six_digit', 'soc_occupation': 'soc_occupation_6digit'})
soc_code_6digit.six_digit = [w.replace('-','')[0:6] for w in soc_code_6digit.six_digit]

data_occupation_output = data_occupation_output.merge(soc_code_4digit, left_on = 'onet_four_digit', right_on = 'four_digit', how = 'left')
data_occupation_output = data_occupation_output.merge(soc_code_6digit, left_on = 'onet_six_digit', right_on = 'six_digit', how = 'left')

# output data
data_occupation_output.to_csv(data_folder + 'data_occupation_agri.csv')








