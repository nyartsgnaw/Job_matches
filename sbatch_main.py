import os 

import pandas as pd 
path_exp = os.path.join(CWDIR,'./experiments/exp_logs.xlsx')
df_exp = pd.read_excel(path_exp)
for i in range(df_exp.shape[0]):
    with open(os.path.join(CWDIR,'./experiments/.idx'),'w') as f:
        idx = f.write(str(i))
    os.system('sbatch submit_job.sh')
    time.sleep(2)