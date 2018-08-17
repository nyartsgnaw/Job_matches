import nltk
import re
#split the data
import pandas as pd 
import json
#import pickle
from nltk.corpus import wordnet as wn
from keras.preprocessing import sequence

from nltk.tokenize import sent_tokenize,word_tokenize

from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

from nltk.corpus import stopwords  
import numpy as np 
import os
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()	

def get_time_series(data,labels,TIME_STEPS=10,START_INDEX=0):

	try:
		data_gen = sequence.TimeseriesGenerator(data, labels,
									length=TIME_STEPS, sampling_rate=1,
									batch_size=1,start_index=int(TIME_STEPS/2))
		
		X = np.array([data_gen[i][0][0] for i in range(len(data_gen))])
		Y = np.array([data_gen[i][1][0] for i in range(len(data_gen))])
	except:
		def TimeseriesGenerator(data,targets,length=10,start_index=0):
			i = start_index 
			X,Y = [],[]
			while i<len(data)-length:
				X.append(data[i:i+length])
#				Y.append(targets[i:i+length])
				Y.append(targets[i])
				i+=1
			X = np.array(X)
			Y = np.array(Y)
			return X,Y
		X,Y = TimeseriesGenerator(data, labels, length=TIME_STEPS, start_index=START_INDEX)		
	return X,Y

def get_similar_words(ls,model,k=20,topn=20,condition = True):
	
	updates = []
	for x in ls:  
		if x in model.wv.vocab.keys():
			#keep ones with high similarity and less ambiguous senses in tree bank
			if condition == True:
				values = [k for k,v in model.wv.most_similar_cosmul(x,topn=topn) if (v >0.95)&(len(wn.synsets(k))<1)]
			else:
				values = [k for k,v in model.wv.most_similar_cosmul(x,topn=topn) if (v >0.95)]

			updates+=values
	output = list(set(updates+ls))
	print(k,len(output),topn,condition)
	k-=1
	if k >=0:
		if len(output)-len(ls) < int(topn/2):
			return output
		
		return get_similar_words(output,model=model,k=k,topn=topn,condition=condition)
	else:
		return output

class prepare_data():

	def __init__(self,section_names_path=os.path.join(CWDIR,'./../tmp/section_names.json')):
		with open(section_names_path, 'r') as f:
			self.section_names = json.load(f)    

		return

	def clean_text(self,text):
		text = re.sub('[\w\.-]+@([\w\.-]+|\.\.\.)', 'A_EMAILADDRESS',text)
		text = re.sub('/|\-', ' ',text)
		text = re.sub('[Zz]ero', '0',text)
		text = re.sub('[Oo]ne', '1',text)
		text = re.sub('[Tt]wo', '2',text)
		text = re.sub('[Tt]hree', '3',text)
		text = re.sub('[Ff]our', '4',text)
		text = re.sub('[Ff]ive', '5',text)
		text = re.sub('[Ss]ix', '6',text)
		text = re.sub('[Ss]even', '7',text)
		text = re.sub('[Ee]ight', '8',text)
		text = re.sub('[Nn]ine', '9',text)
		text = re.sub('[Tt]en', '10',text)

		text = re.sub('(\d+[-\.\s]??\d+[-\.\s]??\d+|\(\d+\)\s*\d+[-\.\s]??\d+|\d+[-\.\s]??\d+)','A_PHONENUMBER',text)
		text = re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+','A_URL', text)
		text = re.sub('(\d{2}(/|-|\.)\w{3}(/|-|\.)\d{4})|([a-zA-Z]+\s\d{2}(,|-|\.|,)?\s\d{4})|(\d{2}(/|-|\.)\d{2}(/|-|\.)\d+)|(\d{2}\s(,|-|\.|,)?[a-zA-Z]+\s\d{4})|\d{4}','A_DATE',text)
		return text



	def clean_pat(self,pat):
		pat = pat.replace('+','\+')
		pat = pat.replace('*','\*')
		pat = pat.replace('?','\?')
		pat = pat.replace('(','\(')
		pat = pat.replace(')','\)')
		return pat


	def split_sections(self,text):
		key = 'ORGANIZATION'
		result = 1
		JD={}
		while True:
			result = re.search('[\r\n](?!=)*[ *[A-Za-z]*]*(?=:)',text)
			if result == None:
				break
			value = text[:result.start()]
			JD[key]=value.strip()
			#update key for next round
			key = result.group().replace('\r\n','').strip()

			text = text[result.end()+1:]
		JD['LEFT']=text
		return JD

	def get_clean_sections(self,texts):
		#split sections
		JD_ls_raw = []
		i = 0
		while i < len(texts):
			#search the company name
			JD_ls_raw.append(self.split_sections(texts[i]))
			i+=1
	#        print(i)

		#merge section names
		JD_ls = JD_ls_raw
		all_names = sum(list(self.section_names.values()),[])
		i=0
		while i < len(JD_ls):
			keys = list(JD_ls[i].keys())
			for k in keys:
				if k in all_names:
					for right_k in self.section_names.keys():
						if k in self.section_names[right_k]:
							JD_ls[i][right_k] = JD_ls[i][k]
							del JD_ls[i][k]            
				else:
	#               print(k)
					del JD_ls[i][k]

			updated_keys = list(JD_ls[i].keys())
			for right_k in list(self.section_names.keys()):
				if right_k not in updated_keys:
					JD_ls[i][right_k] = ''
			i+=1
		return JD_ls




	#concat all nouns
	def lower_sentence_capitalized(self,sentence):
		if sentence =='':
			return sentence    
		k = 0
	#        print('before',sentence[:15])                        
		while (k<len(sentence)-1)&(re.search('[A-Z]',sentence[k])!=None):
			l = 0
			#search for the stop point of this word
			while (k+l+1<len(sentence)-1)&(re.search('\w',sentence[k+l])!=None):
				l+=1
			#if the next word is not capitalized, lower case
			if re.search('[A-Z]',sentence[k+l+1])==None:
				sentence=sentence[:k]+sentence[k].lower()+sentence[k+1:]  
	#               print('after',sentence[:15])
				break                      
			k+=1
		return sentence


	def concat_consecutive_pronoun(self,text):
		if text =='':
			return text
		tokenized = word_tokenize(text)
		i = 0
		pos_dict = {}
		original_l = len(tokenized)
		while i<original_l-1:
			if re.search('[A-Z]',tokenized[i])!=None:
				j = 0
				#doesn't have punctuation OR does have A-Z
				while (re.search('[^\w]',tokenized[i+j])==None) & (re.search('[A-Z]',tokenized[i+j])!=None) & (re.search('[a-z]',tokenized[i+j+1])!=None):
					j+=1
					if i+j+1>len(tokenized)-1:
						break
				#after find the last consective capitalized word
				if j>0:
					pos_dict[i]=i+j
					i = i+j-1
			i+=1

		new = []
		keys = list(pos_dict.keys())
		prev_end = 0
		for i in range(len(keys)):
			start = keys[i]
			end = pos_dict[start]       
			gap_words = tokenized[prev_end:start]
			new_word = ['-'.join(tokenized[start:end])]
			new = new+gap_words+new_word
			prev_end = end
	#        print(start,end,gap_words)
		new+=tokenized[prev_end:]
		return ' '.join(new)


	def concat_pronouns(self,text):
		if text == '':
			return text
		sents = sent_tokenize(text)
		for j in range(len(sents)):
			sents[j] = self.lower_sentence_capitalized(sents[j])
		cleaned_text = ' '.join(sents)
		concatened_text = self.concat_consecutive_pronoun(cleaned_text)
		return concatened_text

	def filter_posTag(self,tagged,pattern = 'NN|VBG'):
		wn_lemmatizer = WordNetLemmatizer()

		j = 0
		qualified_words = []
		while j < len(tagged):
			if re.search(pattern,tagged[j][1])!=None:
				word = wn_lemmatizer.lemmatize(tagged[j][0])
				qualified_words.append(word)
			j+=1
			
		return qualified_words


def remove_stopwords(texts):
	stop_words = list(set(stopwords.words('english')))+'% , ! . + - ( ) : \" \" ; < > { } _ = @ # $ % ^ & * ~ ` [ ] /'.split()+['--','...','']	
	tagged_ls = [word_tokenize(x) for x in texts]
	stoplower_ls = []
	for tkn in tagged_ls:
		#check stop words, and lower the words
		filtered_tkn = [w.lower() for w in tkn if w not in stop_words]
		stoplower_ls.append(filtered_tkn)
	texts = [' '.join(x) for x in stoplower_ls]
	return texts
if __name__ =='__main__':
	job_description_path = os.path.join(CWDIR,'./../tmp/job_description.json')
	model_path = os.path.join(CWDIR,'./../../logs/models/Word2Vec_nouns.model')
	pdata = prepare_data()
	# get data
	path_data = os.path.join(CWDIR,'./../raw_data/online-job-postings/data job posts.csv')
	df_raw = pd.read_csv(path_data)
	df_raw = df_raw.loc[~df_raw['Title'].isna(),]
	texts = df_raw.jobpost.values

	if ~os.path.isfile(job_description_path):
		# prepare data 
		## filter out uninformative tokens
		print('start pre-processing')
		data = []
		for i in range(len(texts)):
			data.append(pdata.clean_text(texts[i]))
		## split sections and merge section names
		print('start spliting sectoins')
		JD_ls = pdata.get_clean_sections(data)
		## concat all pronouns for each merged sections
		print('start concatening pronouns')
		for i in range(len(JD_ls)):
			JD = JD_ls[i]
			for k in JD:
				JD[k] = pdata.concat_pronouns(JD[k])
			JD_ls[i] = JD
		#get the model
		if os.path.isfile(model_path): #load the existed model
			print('loading model')
			model = Word2Vec.load(model_path)
		
		else: #retrain the model
			## select wanted sections
			print('start preparing data for model training')
			data_ls = []
			for i in range(len(JD_ls)):
				for k in ['qualification','skill']:
					data_ls.append(JD_ls[i][k])
			data = ' '.join(data_ls)

			##select nouns by part of speech model
			tokenized_sent = sent_tokenize(data)
			stoplower_ls = [x.split(' ') for x in remove_stopwords(tokenized_sent)]
			tagged_ls = [nltk.pos_tag(x) if x!=[''] else ('','') for x in stoplower_ls]

			pos_tokenized_ls = [pdata.filter_posTag(x) if x!=('','') else [''] for x in tagged_ls]

			# train skip-gram model
			print('start model training')
			model = Word2Vec( size=100, window=7, min_count=1, workers=16)
			model.build_vocab(pos_tokenized_ls)
			model.train(pos_tokenized_ls, total_examples=model.corpus_count, epochs=model.iter)


		# impute education,skills,experience with skip-gram model
		print('start extracting data by model')
		init_ls_skill = [
				'api','adobe','css','javascript','.net','php','mysql','excel','oracle','cad',
				'python','sql','tensorflow','keras','windows','macos', 'ms-office','chinese','spanish','languages','french',
				'tableau', 'latex', 'ggplot', 'd3.js', 'excel','visio', 'ssl', 'sockets', 'perl', 'android', 'web', 'unix',
				'linux','spark','hadoop','scraper','cuda','openmp','mpi','sge','networkx','java','c++','html','aws','gcp','git','c #',
			]
		skill_ls = sorted(list(set(get_similar_words(init_ls_skill,model,k=20,topn=4,condition=True)+init_ls_skill)))

		init_ls_education = ['cpa','master-level','cfa','acca','master','university','','higher-education','mba','phd','bachelor','graduate']
		education_ls = sorted(list(set(get_similar_words(init_ls_education,model,k=3,topn=3,condition=False)+init_ls_education)))

		for i in range(len(JD_ls)):
			JD_ls[i].update({'education':[],'skill_tech':[],'experience':[]})
			for k in ['qualification','skill']:            
				JD_ls[i]['experience']+=re.findall('[0-9]+(?= *year)',JD_ls[i][k])#only check the upper limit for experience
				tokenized = sorted(word_tokenize(JD_ls[i][k]))
				JD_ls[i]['education']+= [x for x in education_ls for y in set(tokenized) if x ==y.lower()]
				JD_ls[i]['skill_tech']+= [x for x in skill_ls for y in set(tokenized) if x ==y.lower()]
				print('JD info extracting:',i)

		#save the results
		model.save(model_path)

		with open(job_description_path, 'w') as f:
			json.dump(JD_ls, f)


	with open(job_description_path,'r') as f:
		JD_ls = json.load(f)
	texts = [','.join(x['skill_tech'])+' '+x['responsibility']+' '+x['qualification'] for x in JD_ls]    
	df_raw['clean_titles'] = remove_stopwords(df_raw['Title'].values)

	df_out = df_raw[['clean_texts','clean_titles']]
	df_out.columns = ['texts','titles']
	path_training_data = os.path.join(CWDIR,'./../df_all.csv')
	df_out.to_csv(path_training_data,index=False)

	count = {}
	for i in range(len(JD_ls)):
		JD = JD_ls[i]
		for k in JD:
			if (JD[k] == '') or (JD[k]==[]):
				try:
					count[k]+=1
				except:
					count[k] = 0
	print('miss counts:')
	print(count)
#*************************************************************************************************
"""
	df_raw['clean_texts'] = remove_stopwords(texts)

	titles = df_raw['Title'].values.copy()
	for i in range(len(titles)):
		t = titles[i]
		#delete what's in the parathes
		if re.search('/',t):
			ls = t.split(' ')
			pos_ls = []
			for i in range(len(ls)):
				#delete single / and add it to previous and mark
				if ls[i] == '/':
					ls[i-1] = ls[i-1]+'/'
					del ls[i]
					pos_ls.append(i-1)
				#mark and delete
				elif re.search('/',ls[i]):
					ls[i] = ls[i].replace('/','')
					print(ls[i])
					pos_ls.append(i)
		
		dict_opt = {}
		for i in range(len(pos_ls)):
			pos = pos_ls[i]
			dict_opt[pos] = [ls[pos],ls[pos+1]]


			
		dict_fix = {0:ls[:pos_ls[0]],len(pos_ls):ls[pos_ls[-1]+2:]}
		i = 0
		while i < len(pos_ls)-1:
			pos = pos_ls[i]
			pos_next = pos_ls[i+1]
			fix = ls[pos+2:pos_next]
			dict_fix[i+1] = fix
			print(fix)
			i+=1
		
		def get_opt(prev,i):
			if i == len(dict_opt):
				return prev
			for k in dict_opt:
				v = dict_opt[k]
				prev = [prev]+v+[dict_fix[i]]
				i+=1
				return get_opt(prev,i)

		get_opt(ls)
		


		for i in range(len(pos_ls)):
			pos = pos_ls[i]		
			opt0,opt1 = dict_opt[pos]

		

		

		piece = ls[0:pos1]
		pos
		fix0 = ls[0:pos1]
		+ pos0
		+ pos0+1
		fix1 = ls[pos1+2:pos2]
		+ pos1
		+ pos1+1
		fix2 = ls[pos2+2:pos3]
		+ pos2
		+ pos2+1
		fix3 = ls[pos3+2:pos4]
		+ pos3
		+ pos3+1
				
			



			i = 0
			for i in range(len(ls)):

				tkn = ls[i]
				if tkn.find('/'):
					tkn = tkn.replace('/','')
				
				ls1 = ls[:i+1]+ls[i+2:]
				ls2 = 


			print(ls)

	#delete what's after /
	#to, on, of, in, at, change orders


		pat_segwords = " to | in | of | for | at | on | - |,"
		if re.search(pat_segwords,t):
			t = re.split(pat_segwords, t)[:2]
			t = t[1] +' '+t[0]
	ls_rep = [
		('\( *\w+.*\w*\)',''), #paratheisis
		('ID *No. *[0-9]+', ''),
		('\w\-[0-9]+',''),
		('[0-9]+','')
	]
	
	def sub_text(text,pat1,pat2):
		if re.search(pat1,text):
			text = re.sub(pat1,pat2,text)
		return text
			



			titles[i] = t
	#replace word descibing job level
	levels = (
		('chair','senior manager'),
		('chairman','senior manager'),
		('scientist','senior analyst'),
		('director','senior manager'),
		('leader','senior manager'),
		('leading','senior'),
		('head','senior manager'),
		(' sr ',' senior '),
		('officer','senior manager'),
		('expert','senior technician'),
		('mid-level','junior'),
		(' jr ',' junior '),
		('intern','entry-level agent'),
		('assitant','entry-level agent'),
		('associate','entry-level'),
		('basic','entry-level'),
		('coordinator','entry-level agent'),
		('support','entry-level'),
		('contractor','short-term agent')
	)
	#replace non-paid, full-time. part-time, short-term, short-trim etc.
	
"""
#*************************************************************************************************
