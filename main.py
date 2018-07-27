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



def start_exp(exp):
	EXP_ID = exp['EXP_ID'] #the name for this experiment 
	MODEL_ID = exp['MODEL_ID'] #model framework
	OUTPUT_DIM = int(exp['OUTPUT_DIM']) # LSTM output vector dimension, should match that of Word2Vec of labels
	INPUT_DIM = int(exp['INPUT_DIM']) # LSTM input vector dimension, length of tokens cut from original data texts for each record
	TIME_STEPS = int(exp['TIME_STEPS']) #for LSTM sequential
	N_EPOCH = int(exp['N_EPOCH']) #for LSTM
	PATIENCE = int(exp['PATIENCE']) #for LSTM
	TRAIN_MODEL = int(exp['IS_TRAIN'])
	LOSS=exp['LOSS_FUNC']
	print(exp)
	

	import_local_package(os.path.join(CWDIR,'./experiments/models/{}.py'.format(MODEL_ID)),[])
	path_vectors = os.path.join(CWDIR,'./logs/models/vectors_JT-{}.csv'.format(INPUT_DIM))
	path_labels = os.path.join(CWDIR,'./tmp/job_titles.txt')
	path_data = os.path.join(CWDIR,'./tmp/df_texts.csv')
	path_model = os.path.join(CWDIR,'./logs/models/LSTM_{}.model'.format(EXP_ID))
	path_eval = os.path.join(CWDIR,'./logs/eval/LSTM_eval_{}.csv'.format(EXP_ID))
	path_training_model = os.path.join(CWDIR,'./logs/LSTM_train_{}.model'.format(EXP_ID))
	path_training_log = os.path.join(CWDIR,'./logs/train_logs/LSTM_logs{}.csv'.format(EXP_ID))
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
	
	labels = pd.read_csv(path_vectors).values
	texts = np.array([x[0] if type(x)!=str else x for x in pd.read_csv(path_data).values])
	# prepare trainig/testing data

	GOOD_i =[]
	for i in range(len(texts)):
		if len(texts[i])>=200:
			GOOD_i.append(i)

	labels = labels[GOOD_i]
	texts = texts[GOOD_i]
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts([' '.join(texts)])
	data = tokenizer.texts_to_sequences(texts)
	data = sequence.pad_sequences(data, padding='post',truncating='post',maxlen=INPUT_DIM) # truncate and pad input sequences
	from keras.preprocessing.sequence import TimeseriesGenerator

	data_gen = TimeseriesGenerator(data, labels,
								length=TIME_STEPS, sampling_rate=1,
								batch_size=1)
	
	X = np.array([data_gen[i][0] for i in range(len(data_gen))])
	Y = np.array([data_gen[i][1] for i in range(len(data_gen))])

	X_train, X_test = train_test_split(X,train_size=0.8)
	Y_train, Y_test = train_test_split(Y,train_size=0.8)




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
		model = train_model(model,X_train=X_train.reshape([-1,1,TIME_STEPS,INPUT_DIM]),\
							Y_train=Y_train,\
							verbose=1,n_epoch=N_EPOCH,validation_split=0,patience=PATIENCE,
							model_path=path_training_model,
							log_path=path_training_log)
		model.save(path_model)

	#evaluate the model
	all_percs = [] #rank of true label in the queue of sorted job titles by cosine similarity
	all_top10 = [] #top10 job titles prediction 
	i= 0 
	while i < len(X_test):
#	while i < 2:
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

	df_log = pd.read_csv(path_training_log)
	try:
		exp['RANK_SCORE'] = np.mean(all_percs)
	except Exception as e:
		print(e)
	try:
		exp['QUIT_LOSS'] = df_log.iloc[-1]['loss']
		exp['QUIT_EPOCH'] = df_log.iloc[-1]['epoch']
		exp['QUIT_MSE'] = df_log.iloc[-1]['mean_squared_error']
	except Exception as e:
		print(e)

	try:
		exp['N_PARAMS'] = model.count_params()
	except Exception as e:
		print(e)
	return exp

if __name__ == '__main__':
	# inputs
	path_exp = os.path.join(CWDIR,'./experiments/exp_logs.xlsx')
	df_exp = pd.read_excel(path_exp)
#	idx =0 
	with open(os.path.join(CWDIR,'./experiments/.idx'),'r') as f:
		idx = int(f.read())
	
	exp = df_exp.iloc[idx]
	if int(exp['IS_RUN'])==1:
		exp = start_exp(exp)
		exp['IS_RUN'] = 0
		df_exp.iloc[idx] = exp
		df_exp.to_excel(path_exp,index=False)


