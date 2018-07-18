

import os
import re 
import pandas as pd 

vector_dim = 100
path_model = './../models/job_title_fasttext'
path_data = './../tmp/job_titles.txt'
path_vector_JD = './../models/vectors_JT.txt'
path_output_csv = './../models/vectors_JT.csv'

if ~os.path.isfile(path_model+'.bin'):
    command = "./../../fastText-0.1.0/fasttext skipgram -input {} -output {} -lr {} -epoch {} -dim {} -minCount {} -maxn {}".format(path_data,path_model,0.05,50,vector_dim,1,10)
    os.system(command)

command = " ./../../fastText-0.1.0/fasttext print-sentence-vectors {} < {} > {}".format(path_model+'.bin',path_data,path_vector_JD)
os.system(command)
    
with open(path_data,'r') as f:
    titles = [x.replace('\n','') for x in f.readlines()]


with open(path_vector_JD,'r') as f:
    raw_vectors = [x.replace('\n','') for x in f.readlines()]


vectors_JT = []
i = 0
while i < len(raw_vectors):
    ls = raw_vectors[i].split()
    j = 0
    while j < len(ls):
        if (re.search('^\-?[0-9]*\.?[0-9]+$',ls[j])!=None) & (re.search('[a-zA-Z]+',ls[j+1])==None):
            word = ' '.join(ls[:j])
            nums = [float(x) for x in ls[j:]]
            if len(nums)>100:
                nums = nums[1:]
            vectors_JT.append(nums)
            break

        j+=1
    i+=1


df = pd.DataFrame(vectors_JT)

df.to_csv(path_output_csv,index=False)