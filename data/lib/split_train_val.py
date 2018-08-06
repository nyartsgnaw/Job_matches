import pandas as pd 
import os
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()	

path_training_data = os.path.join(CWDIR,'./../df_all.csv')
df_all = pd.read_csv(path_training_data).dropna()

long_is =[]
for i in range(df_all.shape[0]):
    text = df_all.iloc[i]['texts']
    if len(text) > 200:
        long_is.append(i)

df_all = df_all.iloc[long_is]


from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(range(df_all.shape[0]),train_size=0.8,random_state=100)


df_all['split'] = ['val' if i in test_idx  else 'train' for i in range(df_all.shape[0])] 



df_all['is_engineer'] = [1 if ('engineer' in x.split(' ') or ('developer' in x.split(' '))) else 0 for x in df_all.titles]

df_all.to_csv(path_training_data,index=False)