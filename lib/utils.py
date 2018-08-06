import operator
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import pandas as pd
import numpy as np 
import re
import os
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()	
def train_model(model,\
		X_train,Y_train,\
		n_epoch =  200,\
		monitor='loss',\
		min_delta=0.0005,\
		patience=5,\
		verbose=2,\
		mode='auto',\
		shuffle = True,\
		validation_split=0.2,\
		log_path = '',\
		model_path = ''):
	#train a model
	#parameter names are the same as keras model.fit

	earlyStopping = EarlyStopping(monitor=monitor,
								min_delta = min_delta,
								patience=patience,
								verbose=verbose,
								mode=mode)
	callbacks = [earlyStopping]
	if log_path != '':
		csv_logger = CSVLogger(log_path,append=True)
		callbacks.append(csv_logger)
	if model_path != '':
		checkpointer = ModelCheckpoint(filepath=model_path, save_best_only=True)
		callbacks.append(checkpointer)

	batch_size = int(np.ceil(len(X_train)/100.0)) # variable batch size depending on number of data points
	mod = model.fit(X_train, Y_train,
					batch_size=batch_size,
					epochs = n_epoch,
					verbose=2,
#						callbacks=[earlyStopping,checkpointer,csv_logger],
					callbacks=callbacks,
					shuffle=shuffle, validation_split=validation_split)
	return mod.model


def load_model(model,path_parameter):
	#load model by weights, to avoid bugs relating to initializing keras computing graph when using keras.models.load_model
	model.load_weights(path_parameter,by_name=True)
	return model

def load_embedding_fasttext(path):
	import fasttext
	model = fasttext.load_model(path, encoding='utf-8')
	path = path.replace('.bin','')

	embedding_matrix = np.zeros((len(model.words) + 1, 200))
	words = list(model.words)
	for i in range(len(model.words)):
		word = words[i]
		embedding_vector = model[word]
		if embedding_vector is not None:
			# words not found in embedding index will be all-zeros.
			embedding_matrix[i] = embedding_vector 

	return embedding_matrix


def load_embedding():
	# load embedding weights, deprecated
	from gensim.models import Word2Vec
	model = Word2Vec.load(os.path.join(CWDIR,'./../embedding/peptideEmbedding.bin'))
	embedding_weights = np.zeros((model.wv.syn0.shape[0]+1,model.wv.syn0.shape[1]))
	for i in range(len(model.wv.vocab)):
		embedding_vector = model.wv[model.wv.index2word[i]]
		if embedding_vector is not None:
			embedding_weights[i] = embedding_vector
	return embedding_weights



def get_rank_df(yhat,titles_test,Y,titles_all):
	from sklearn.metrics.pairwise import cosine_similarity
	sim_score = cosine_similarity(yhat,Y)
	outputs = []
	for idx in range(sim_score.shape[0]):
		rank_dict = {}
		for j in range(sim_score.shape[1]):
			rank_dict[idx,j] = sim_score[idx][j]
		sorted_dict = sorted(rank_dict.items(),key=operator.itemgetter(1),reverse=True)
		rank_seqs = []
		i = 0
		print()
		print('Job:',titles_test[idx])
		while i <len(sorted_dict):
			k,v = sorted_dict[i]
			idx,j =k
			if i<10:
				print('  ',titles_all[j])
				rank_seqs.append(titles_all[j])
			if j == idx:
				rank = i/len(sorted_dict)
				print('Percentage_Rank of {}: {}'.format(idx,rank))
				break
			i+=1
		output = [titles_test[idx]]+[rank] + rank_seqs
		outputs.append(output)
	df = pd.DataFrame(outputs,columns=['label']+['rank']+list(range(10)))
	return df