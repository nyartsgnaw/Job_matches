from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np 
from sklearn.model_selection import train_test_split
import pandas as pd
import json
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
import_local_package(os.path.join(CWDIR,'./lib/utils.py'),['train_model','load_model','get_rank_info','predict_cosine_similarity'])
import_local_package(os.path.join(CWDIR,'./lib/prepare_data.py'),[])


if __name__ == '__main__':
	# inputs
	df_exp = pd.read_excel(os.path.join(CWDIR,'./experiments/exp_logs.xlsx'))
	idx = 11
	exp = df_exp.iloc[idx]
	EXP_ID = exp['EXP_ID'] #the name for this experiment 
	MODEL_ID = exp['MODEL_ID'] #model framework
	OUTPUT_DIM = int(exp['OUTPUT_DIM']) # LSTM output vector dimension, should match that of Word2Vec of labels
	INPUT_DIM = int(exp['INPUT_DIM']) # LSTM input vector dimension, length of tokens cut from original data texts for each record
	TIME_STEPS = int(exp['TIME_STEPS']) #for LSTM sequential
	N_EPOCH = int(exp['N_EPOCH']) #for LSTM
	PATIENCE = int(exp['PATIENCE']) #for LSTM
	TRAIN_MODEL = int(exp['TRAIN_MODEL'])
	LOSS=exp['LOSS']
	print(exp)
	

	import_local_package(os.path.join(CWDIR,'./lib/models/{}.py'.format(MODEL_ID)),[])
	path_vectors = os.path.join(CWDIR,'./logs/models/vectors_JT-{}.csv'.format(INPUT_DIM))
	path_labels = os.path.join(CWDIR,'./tmp/job_titles.txt')
	path_data = os.path.join(CWDIR,'./tmp/job_description.json')
	path_model = os.path.join(CWDIR,'./logs/models/LSTM_{}.model'.format(EXP_ID))
	path_eval = os.path.join(CWDIR,'./logs/eval/LSTM_eval_{}.csv'.format(EXP_ID))
	os.system('mkdir -p {}'.format(os.path.join(CWDIR,'./logs/eval/')))
	# fix random seed for reproducibility
	np.random.seed(7)
	# get embeddings
	with open(path_labels,'r') as f:
		titles = [x.replace('\n','') for x in f.readlines()]

	if not os.path.isfile(path_vectors):
		print('Warning: fasfttext model wasn\'t found, start retraining...')
		path_fasttext = os.path.join(CWDIR,'./../fastText-0.1.0/fasttext')
		path_model_fasttext = os.path.join(CWDIR,'./logs/models/job_title_fasttext')
		import_local_package(os.path.join(CWDIR,'./lib/train_fasttext.py'),['train_fasttext','train_fasttext2'])
		if os.path.isfile(path_fasttext):
			print('Start training fasttext in Linux environment...')
			df_vectors,labels = train_fasttext2(titles,path_model = path_model_fasttext ,vector_dim=OUTPUT_DIM,path_fasttext=path_fasttext,path_output_csv=path_vectors)
		else:
			print('Start training fasttext in Python environment...')
			model_fasttext = train_fasttext(titles,path_model = path_model_fasttext)
			labels = np.array([np.array(model_fasttext[word]) for word in titles])
	
	df_vectors = pd.read_csv(path_vectors)
	labels = df_vectors.values
	# prepare trainig/testing data
	with open(path_data,'r') as f:
		JD_ls = json.load(f)
	texts = [x['responsibility']+' '+x['qualification'] for x in JD_ls]    
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts([' '.join(texts)])
	data = tokenizer.texts_to_sequences(texts)
	data = sequence.pad_sequences(data, padding='post',truncating='post',maxlen=INPUT_DIM) # truncate and pad input sequences
	X_train, X_test = train_test_split(data,train_size=0.8)
	Y_train, Y_test = train_test_split(labels,train_size=0.8)

	# create the model    
	from keras.models import load_model
	if os.path.isfile(path_model):
		model = load_model(path_model)
	else:
		embedding_matrix = []
#        embedding_matrix = load_embedding_fasttext(path_JD)
 #       model = create_LSTM(input_dim=INPUT_DIM,output_dim=OUTPUT_DIM,embedding_matrix=embedding_matrix)
		model = create_LSTM(input_dim=INPUT_DIM,output_dim=OUTPUT_DIM,time_steps=TIME_STEPS,embedding_matrix=embedding_matrix)
	if TRAIN_MODEL == True:
		from keras.optimizers import SGD, Adam, RMSprop
		adam=Adam(lr=0.005, beta_1=0.9 ,decay=0.001)
		model.compile(loss=LOSS, optimizer=adam, metrics=['mse'])
		model = train_model(model,X_train=X_train.reshape([-1,INPUT_DIM,TIME_STEPS,1]),\
							Y_train=Y_train,\
							verbose=1,n_epoch=N_EPOCH,validation_split=0,patience=PATIENCE,
							model_path=os.path.join(CWDIR,'./logs/LSTM_train_{}.model'.format(EXP_ID)),
							log_path=os.path.join(CWDIR,'./logs/train_logs/LSTM_logs{}.csv'.format(EXP_ID)))
		model.save(path_model)

	#evaluate the model
	all_percs = [] #rank of true label in the queue of sorted job titles by cosine similarity
	all_top10 = [] #top10 job titles prediction 
	i= 0 
	while i < len(X_test):
		print(i)
		all_percs.append(get_rank_info(model,i,X_test,np.concatenate([Y_test,Y_train]))['rank_idx_correct'])
		all_top10.append(get_rank_info(model,i,X_test,np.concatenate([Y_test,Y_train]))['top10'])
		i+=1    
	ls1 = []
	for x in all_top10:
		for k,v in x.items():
			ls1.append([k]+list(v))
			
	df = pd.concat([pd.DataFrame(ls1,columns=['label']+list(range(10))),pd.DataFrame(all_percs, columns=['rank_percentage'])],axis=1)
	df.to_csv(path_eval,index=False)

	#model.save('./../logs/models/LSTM1-Data_nouns-Epoch_100-0.4983134954955406.model')
	#model.save('./../logs/models/LSTM1-Data_nouns-Epoch_100-0.4983134954955406.model')
	#model.save('./../logs/models/LSTM2-Data_all-Epoch_100-0.5001153382010389.model',overwrite=True,include_optimizer=True)

