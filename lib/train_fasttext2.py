

import re 
import pandas as pd 
import os
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()	
vector_dim = 100
path_model = os.path.join(CWDIR,'./../models/job_title_fasttext')
path_data = os.path.join(CWDIR,'./../tmp/job_titles.txt')
path_vector_JD = os.path.join(CWDIR,'./../models/vectors_JT.txt')
path_output_csv = os.path.join(CWDIR,'./../models/vectors_JT-100.csv')
path_fasttext = os.path.join(CWDIR,'./../../fastText-0.1.0/fasttext')


if ~os.path.isfile(path_model+'.bin'):
    command = "{} skipgram -input {} -output {} -lr {} -epoch {} -dim {} -minCount {} -maxn {}".format(path_fasttext,path_data,path_model,0.05,50,vector_dim,1,10)
    os.system(command)

command = "{} print-sentence-vectors {} < {} > {}".format(path_fasttext,path_model+'.bin',path_data,path_vector_JD)
os.system(command)
    
with open(path_data,'r') as f:
    titles = [x.replace('\n','') for x in f.readlines()]


with open(path_vector_JD,'r') as f:
    raw_vectors = [x.replace('\n','') for x in f.readlines()]


vectors_JT = []
i = 0
while i < len(raw_vectors):
    ls = raw_vectors[i].split()

    word = ' '.join(ls[:-vector_dim])
    nums = [float(x) for x in ls[-vector_dim:]]
    if len(nums)<vector_dim:
        print(ls)
    if len(nums)>vector_dim:
        nums = nums[1:]
    vectors_JT.append(nums)

    i+=1


df = pd.DataFrame(vectors_JT)

df.to_csv(path_output_csv,index=False)
