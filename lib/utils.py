import operator
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from nltk.corpus import wordnet as wn
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

with open(os.path.join(CWDIR,'./../tmp/job_titles.txt'),'r') as f:
	titles = [x.replace('\n','') for x in f.readlines()]


from sklearn.metrics.pairwise import cosine_similarity
def predict_cosine_similarity(model,idx,X,Y):
	yhat = model.predict(X[idx].reshape([-1,X.shape[1],1,1]))
	return cosine_similarity(yhat,Y)



def get_rank_info(model,idx,X,Y):
	sim_score = predict_cosine_similarity(model,idx,X,Y)
	rank_dict = {}
	for i in range(len(sim_score[0])):
		rank_dict[i] = sim_score[0][i]

	sorted_dict = sorted(rank_dict.items(),key=operator.itemgetter(1),reverse=True)
	
	ranks = []
	print()
	print('Job:',titles[idx])
	for k,v in sorted_dict[:10]:
		print('  ',titles[k])
		ranks.append(titles[k])

	i = 0
	while i <len(sorted_dict):
		k,v = sorted_dict[i]
		if k == idx:
			print('LABEL_AT_RANKING_PERCENTAGE:',i/len(sorted_dict))
			break
		i+=1
	return {'rank_idx_correct':i/len(sorted_dict),'top10':{titles[idx]:ranks}}


