# train a fasttext model to map Job Titles into vector representation
from prepare_data import prepare_data


def train_fasttext(texts,path_model,vector_dim=200):
	pdata = prepare_data()
	labels = []
	for i in range(len(texts)):
		tmp = pdata.clean_text(texts[i])
		tmp =tmp.replace('\n','')
		tmp = tmp.replace('\r','')
		tmp = ' '.join(tmp.split())
		labels.append(tmp)
	
	path_data = './../tmp/tmp.txt'


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
		command = "./../../fastText-0.1.0/fasttext skipgram -input {} -output {} -lr {} -epoch {} -dim {} -minCount {} -maxn {}".format(path_data,path_model,0.05,50,vector_dim,1,10)
		os.system(command)
	return model_fasttext
	
if __name__ == '__main__':
	#get data (the job description titles)
	import pandas as pd  
	df_raw = pd.read_csv('./../data/online-job-postings/data job posts.csv')
	df_raw = df_raw.loc[~df_raw['Title'].isna(),]	
	titles = df_raw.Title.values
	#train model
	model_fasttext = train_fasttext(titles,path_model = './../models/job_title_fasttext')

	# evaluate fasttext model
	import sklearn
	sklearn.metrics.pairwise.cosine_similarity([model_fasttext['Programs Manager']],[model_fasttext['Software Developer']])
	sklearn.metrics.pairwise.cosine_similarity([model_fasttext['Programs Manager']],[model_fasttext['Project Manager']])
	sklearn.metrics.pairwise.cosine_similarity([model_fasttext['Data engineer']],[model_fasttext['Software Developer']])
	sklearn.metrics.pairwise.cosine_similarity([model_fasttext['Data engineer']],[model_fasttext['Data Developer']])
	sklearn.metrics.pairwise.cosine_similarity([model_fasttext['Data engineer']],[model_fasttext['Hadoop developer']])
	sklearn.metrics.pairwise.cosine_similarity([model_fasttext['Data scientist']],[model_fasttext['Accountant/Financial Analyst']])




