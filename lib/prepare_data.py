import nltk
import re
#split the data
import pandas as pd 
import json
#import pickle

from nltk.tokenize import sent_tokenize,word_tokenize

from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

from nltk.corpus import stopwords  

import os
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()	

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



if __name__ =='__main__':
    job_description_path = os.path.join(CWDIR,'./../tmp/job_description.json')
    model_path = os.path.join(CWDIR,'./../logs/models/Word2Vec_nouns.model')

    pdata = prepare_data()
    # get data
    df_raw = pd.read_csv(os.path.join(CWDIR,'./../data/online-job-postings/data job posts.csv'))
    df_raw = df_raw.loc[~df_raw['Title'].isna(),]
    texts = df_raw.jobpost.values
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
        tagged_ls = [nltk.pos_tag(word_tokenize(x)) for x in tokenized_sent]
        pos_tokenized_ls = [filter_posTag(x) for x in tagged_ls]

        ##cut off stopwords
        stop_words = list(set(stopwords.words('english')))+[x for x in '%,!.+-():\"\";<>\{\}_=@#$%^&*~`\[\]/\\']+['--','...','']
        
        
        stoplower_ls = []
        for tkn in pos_tokenized_ls:
            #check stop words, and lower the words
            filtered_tkn = [w.lower() for w in tkn if w not in stop_words]
            stoplower_ls.append(filtered_tkn)

        tagged_ls = [nltk.pos_tag(x) for x in stoplower_ls]
        pos_tokenized_ls = [pdata.filter_posTag(x) for x in tagged_ls]
        
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