from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np 
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import train_model,load_model,get_rank_info,predict_cosine_similarity
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


if __name__ == '__main__':
    OUTPUT_DIM = 100
    INPUT_DIM = 200
    TIME_STEPS = 1
    # fix random seed for reproducibility
    np.random.seed(7)
    # get embeddings
    with open(os.path.join(CWDIR,'./../tmp/job_titles.txt'),'r') as f:
        titles = [x.replace('\n','') for x in f.readlines()]
    
    path_output_csv = os.path.join(CWDIR,'./../models/vectors_JT.csv')
    df_vectors = pd.read_csv(path_output_csv)

    labels = df_vectors.values

    job_description_path = os.path.join(CWDIR,'./../tmp/job_description.json')
    with open(job_description_path,'r') as f:
        JD_ls = json.load(f)

    texts = [x['responsibility']+' '+x['qualification'] for x in JD_ls]    

    # prepare trainig/testing data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([' '.join(texts)])
    data = tokenizer.texts_to_sequences(texts)
    
    vocab_size = len(tokenizer.word_index) + 1 # determine the vocabulary size
    data = sequence.pad_sequences(data, padding='post',truncating='post',maxlen=INPUT_DIM) # truncate and pad input sequences
    # split train/test
    X_train, X_test = train_test_split(data,train_size=0.8)
    Y_train, Y_test = train_test_split(labels,train_size=0.8)

    # create the model    
    import_local_package(os.path.join(CWDIR,'./models/LSTM_3.py'),[])
    embedding_matrix = []
#    embedding_matrix = load_embedding_fasttext(path_JD)
#    model = create_LSTM(input_dim=INPUT_DIM,output_dim=OUTPUT_DIM,time_steps=TIME_STEPS,embedding_matrix=embedding_matrix)
    model = create_LSTM(input_dim=INPUT_DIM,output_dim=OUTPUT_DIM,embedding_matrix=embedding_matrix)
    model.compile(loss='cosine_proximity', optimizer='adam', metrics=['mse'])

    model = train_model(model,X_train=X_train.reshape([-1,INPUT_DIM,TIME_STEPS,1]),Y_train=Y_train,verbose=1,n_epoch=200,validation_split=0,patience=20,model_path='LSTM.model',log_path='LSTM_logs.csv')

    all_percs = [] #rank of true label in the queue of sorted job titles by cosine similarity
    all_top10 = [] #top10 job titles prediction 
    i= 0 
    while i < len(X_test):
        print(i)
        all_percs.append(get_rank_info(model,i,X_test,np.concatenate([Y_test,Y_train]))['rank_idx_correct'])
        all_top10.append(get_rank_info(model,i,X_test,np.concatenate([Y_test,Y_train]))['top10'])
        i+=1    
    

    #model.save('./../models/LSTM1-Data_nouns-Epoch_100-0.4983134954955406.model')
    #model.save('./../models/LSTM1-Data_nouns-Epoch_100-0.4983134954955406.model')
    #model.save('./../models/LSTM2-Data_all-Epoch_100-0.5001153382010389.model',overwrite=True,include_optimizer=True)
    model.save(os.path.join(CWDIR,'./../models/new.model'))