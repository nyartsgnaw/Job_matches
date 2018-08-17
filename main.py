
import numpy as np 
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import pandas as pd
import json
import os
import datetime
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()	

from keras import metrics
from keras.optimizers import SGD, Adam, RMSprop
pd.options.mode.chained_assignment = None  # default='warn'

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
import_local_package(os.path.join(CWDIR,'./lib/utils.py'),['train_model','load_model','get_rank_df'])
import_local_package(os.path.join(CWDIR,'./data/lib/prepare_data.py'),[])



def start_exp(exp):
	# setup model parameters
	start_time = datetime.datetime.now()
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

	# setup logging address
	path_vectors = os.path.join(CWDIR,'./logs/models/vectors_JT-{}.csv'.format(INPUT_DIM))
	path_model = os.path.join(CWDIR,'./logs/models/LSTM_{}.model'.format(EXP_ID))
	path_eval = os.path.join(CWDIR,'./logs/eval/LSTM_eval_{}.csv'.format(EXP_ID))
	path_training_model = os.path.join(CWDIR,'./logs/models/LSTM_train_{}.model'.format(EXP_ID))
	path_training_log = os.path.join(CWDIR,'./logs/train_logs/LSTM_logs{}.csv'.format(EXP_ID))
#	if os.path.isfile(path_training_log):
#		os.remove(path_training_log)
	os.system('mkdir -p {}'.format(os.path.join(CWDIR,'./logs/eval/')))
	os.system('mkdir -p {}'.format(os.path.join(CWDIR,'./logs/models/')))
	os.system('mkdir -p {}'.format(os.path.join(CWDIR,'./logs/train_logs/')))
	# fix random seed for reproducibility
	np.random.seed(7)

	# read the label Ys
	path_vectors = os.path.join(CWDIR,'./logs/models/vectors_JT-{}.csv'.format(INPUT_DIM))
	if not os.path.isfile(path_vectors):
		os.system('python {}'.format(os.path.join(CWDIR,'./lib/train_fasttext.py')))

	labels = pd.read_csv(path_vectors).values

	# read the data Xs
	path_data = os.path.join(CWDIR,'./data/df_all.csv')
	df_all = pd.read_csv(path_data)

	# encode the data into sequence
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts([' '.join(df_all['texts'])])
	data = tokenizer.texts_to_sequences(df_all['texts'])
	data = sequence.pad_sequences(data, padding='post',truncating='post',maxlen=INPUT_DIM) # truncate and pad input sequences

	# prepare trainig/testing data/labels
	judge = (df_all['split']=='train').values
	train_data = data[judge]
	test_data = data[~judge]
	train_labels = labels[judge]
	test_labels = labels[~judge]

	# serialize the data/labels to model input/output format
	X_train, Y_train = get_time_series(train_data,train_labels,TIME_STEPS,0)
	X_test, Y_test = get_time_series(test_data,test_labels,TIME_STEPS,0)

	

	# create/load the model    
	from keras.models import load_model
	if os.path.isfile(path_model):
		model = load_model(path_model)
	else:
		embedding_matrix = []
#        embedding_matrix = load_embedding_fasttext(path_JD)
 #       model = create_LSTM(input_dim=INPUT_DIM,output_dim=OUTPUT_DIM,embedding_matrix=embedding_matrix)
		model = create_LSTM(input_dim=INPUT_DIM,output_dim=OUTPUT_DIM,time_steps=TIME_STEPS,embedding_matrix=embedding_matrix)
	# train the model
	if TRAIN_MODEL == True:
		adam=Adam(lr=0.005, beta_1=0.9 ,decay=0.001)
		model.compile(loss=LOSS, optimizer=adam, metrics=['mse','cosine_proximity'])
		model = train_model(model,X_train=X_train.reshape([-1,1,TIME_STEPS,INPUT_DIM]),\
							Y_train=Y_train,\
							verbose=1,n_epoch=N_EPOCH,validation_split=0.1,patience=PATIENCE,\
							model_path=path_training_model,
							log_path=path_training_log)
		model.save(path_model)
	



	# evaluate the model by ranking percentage score
	yhat = model.predict(X_test.reshape([-1,1,TIME_STEPS,INPUT_DIM]))
	# prepare the original testing labels
	titles_all = df_all.titles.values
	titles_test = titles_all[~judge]
	Y = np.concatenate([Y_test,Y_train])

	df = get_rank_df(yhat,titles_test,Y,titles_all)
#	df = get_rank_df(yhat,titles_test,Y_test,titles_test)
	df.to_csv(path_eval,index=False)

	# log the results
	exp['start_time'] = str(start_time.replace(microsecond=0))
	exp['end_time'] = str(datetime.datetime.now().replace(microsecond=0))

	try:
		exp['RANK_SCORE'] = df['rank'].mean()
	except Exception as e:
		print(e)
	try:
		df_log = pd.read_csv(path_training_log)
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
		df_exp = pd.read_excel(path_exp) #re-read the dataframe because multiple threads are handling the program simultaneously 
		df_exp.iloc[idx] = exp
		df_exp.to_excel(path_exp,index=False)


