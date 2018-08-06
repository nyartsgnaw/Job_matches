# train a fasttext model to map Job Titles into vector representation
import pandas as pd
import os
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()	
def import_local_package(addr_pkg,function_list=[]):
	#import local package by address
	#it has to be imported directly in the file that contains functions required the package, i.e. it cannot be imported by from .../utils import import_local_package
	import importlib.util
	spec = importlib.util.spec_from_file_location('pkg', addr_pkg)
	myModule = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(myModule)
	if len(function_list)==0:
		import re
		function_list = [re.search('^[a-zA-Z]*.*',x).group() for x in dir(myModule) if re.search('^[a-zA-Z]',x) != None]

	for _f in function_list:
		try:
			eval(_f)
		except NameError:
			exec("global {}; {} = getattr(myModule,'{}')".format(_f,_f,_f)) #exec in function has to use global in 1 line
			print("{} imported".format(_f))

	return
import_local_package(os.path.join(CWDIR,'./../data/lib/prepare_data.py'),['prepare_data'])


def train_fasttext(texts,path_model,vector_dim=200):
	pdata = prepare_data()
	labels = []
	for i in range(len(texts)):
		tmp = pdata.clean_text(texts[i])
		tmp =tmp.replace('\n','')
		tmp = tmp.replace('\r','')
		tmp = ' '.join(tmp.split())
		labels.append(tmp)
	
	path_data = os.path.join(CWDIR,'./../data/tmp/tmp.txt')
	path_model = path_model.replace('.bin','')
	

	with open(path_data,'w') as f:
		i=0
		while i <len(labels):
			f.write(labels[i]+'\n')
			i+=1

	# train fasttext
	try:
		import fasttext
		model_fasttext = fasttext.skipgram(path_data, path_model, lr=0.05,epoch=50, dim=vector_dim,min_count=1,maxn=10,silent=0)
	except:
		command = "{} skipgram -input {} -output {} -lr {} -epoch {} -dim {} -minCount {} -maxn {}".format(path_fasttext,path_data,path_model,0.05,50,vector_dim,1,10)
		os.system(command)
	os.remove(path_data)
	return model_fasttext
VECTOR_DIM = 200
def train_fasttext2(texts,path_model,vector_dim=VECTOR_DIM,path_fasttext='./../../fastText-0.1.0/fasttext',path_output_csv = os.path.join(CWDIR,'./../logs/models/vectors_JT-{}.csv'.format(VECTOR_DIM))):
#   vector_dim = 200
#   path_model = os.path.join(CWDIR,'./../models/job_title_fasttext')
#   path_fasttext = os.path.join(CWDIR,'./../../fastText-0.1.0/fasttext')
	path_data = os.path.join(CWDIR,'./../data/tmp/job_titles.txt')
	path_vector_JD = os.path.join(CWDIR,'./../logs/models/vectors_JT-{}.txt'.format(vector_dim))
	path_model = path_model.replace('.bin','')

	pdata = prepare_data()
	labels = []
	for i in range(len(texts)):
		tmp = pdata.clean_text(texts[i])
		tmp =tmp.replace('\n','')
		tmp = tmp.replace('\r','')
		tmp = ' '.join(tmp.split())
		labels.append(tmp)
	with open(path_data,'w') as f:
		i=0
		while i <len(labels):
			f.write(labels[i]+'\n')
			i+=1


	if ~os.path.isfile(path_model+'.bin'):
		command = "{} skipgram -input {} -output {} -lr {} -epoch {} -dim {} -minCount {} -maxn {}".format(path_fasttext,path_data,path_model,0.05,50,vector_dim,1,10)
		os.system(command)

	command = "{} print-sentence-vectors {} < {} > {}".format(path_fasttext,path_model+'.bin',path_data,path_vector_JD)
	os.system(command)
		


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

	df_vectors = pd.DataFrame(vectors_JT)

	df_vectors.to_csv(path_output_csv,index=False)
	os.remove(path_vector_JD)

	return df_vectors,labels


if __name__ == '__main__':
	#get data (the job description titles)
	import pandas as pd  
	df_raw = pd.read_csv(os.path.join(CWDIR,'./../data/online-job-postings/data job posts.csv'))
	df_raw = df_raw.loc[~df_raw['Title'].isna(),]	
	titles = df_raw.Title.values
	#train model
	"""
	model_fasttext = train_fasttext(titles,path_model = os.path.join(CWDIR,'./../logs/models/job_title_fasttext'))
	import sklearn
	sklearn.metrics.pairwise.cosine_similarity([model_fasttext['Programs Manager']],[model_fasttext['Software Developer']])
	sklearn.metrics.pairwise.cosine_similarity([model_fasttext['Programs Manager']],[model_fasttext['Project Manager']])
	sklearn.metrics.pairwise.cosine_similarity([model_fasttext['Data engineer']],[model_fasttext['Software Developer']])
	sklearn.metrics.pairwise.cosine_similarity([model_fasttext['Data engineer']],[model_fasttext['Data Developer']])
	sklearn.metrics.pairwise.cosine_similarity([model_fasttext['Data engineer']],[model_fasttext['Hadoop developer']])
	sklearn.metrics.pairwise.cosine_similarity([model_fasttext['Data scientist']],[model_fasttext['Accountant/Financial Analyst']])
	"""

	df_vectors,labels = train_fasttext2(titles,path_model = os.path.join(CWDIR,'./../logs/models/job_title_fasttext'),vector_dim=200,path_fasttext='./../../fastText-0.1.0/fasttext')

	# evaluate fasttext model




