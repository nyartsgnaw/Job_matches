# Job_matches
Goal: match job description to job titles


Here are the workfolow I'll follow, the red part are parts I've already done.
0. Data collection:
0.1. by Kaggle:https://www.kaggle.com/madhab/jobposts  
0.2. by Scrapper (require extra effort to customize on job description   scraping task).  
0.3. by emailed scraped dataset (skipped for now).  

#### 1. Cleaning:
1.1. split Job Description in sections <TITLE,LOCATION,DESCRIPTION,RESPONSIBILITIES,QUALIFICATIONS,ABOUT，EXPERIENCE，EDUCATION>.  
1.2. replace distracted content <url,email_address,phone_or_fax_number, date, abbreviations>.  
1.3. build a lookup table to uniform section names with slightly different spelling describing the same thing (e.g. "JOB DESCRIPTION" vs "DESCRIPTION").   
1.4. remove leakage content <job_title>.    

#### 2.Model input feature extracting:
2.1. extract informative content <location, organization, skills(both technical and soft skills), experience_year_requirement>.  

##### location & organization recognition
+ take location/organization from the section <LOCATION, ORGANIZATION>.  
+ the location could be uniformly formatted to CITY/STATES by Google geocoding API (skipped).  

##### skill entity recognition:
+ part of speech tagging to extract nouns.  
+ merge tokens: lowercase, stemming, lemmatization, concat all consective pro-nouns (e.g. United States -> United-States).  
+ remove stopwords, punctuations.  
+ SkipGram with the window size of the length of 1 job description section to get a skip-gram model M.  
+ list some skills I am familiar with, predict nearest neighbors of these skills in the SkipGram map and select only those that are POS tagged nouns and has 0 sense in treebank (word like unique names).  
+ double check the list by scraping Wikipedia (skipped).  

2.2. get the model input feature set which contains sections <DESCRIPTION, RESPONSIBILITIES, QUALIFICATION> and ignores other sections.  


#### 3. Modeling: 

3.1. Get vector representation for Job Description.  

##### build skills SkipGram model
+ locate skills by skills entity recognition in 2.1.  
+ train a SkipGram model.  

##### concat below features
+ LSTM(DESCRIPTION_text)  
+ LSTM(RESPONSIBILITIES_text)  
+ LSTM(QUALIFICATION_text)  
+ average(SkipGram(skills))  
+ experience_year_requirement  

+ get Vector A: feed the concatened feature to a Neural Networks and get an output vector of 200 dimension.  

3.2. Get vector representation for Job Title (dimension 200).  

+ get Vector B:SkipGram(Job_Titles).  

3.3. train Neural Network parameters by minimizing cosine similarity between Vector A and Vector B.  


#### 4. Evaluating:
4.1. train on 1/2 of the data, validate on 1/4, test on 1/4
4.2. for each testing Job Description X, get its output from neural network, calculate output cosine similarities with all SkipGram(Job Title), select the best as prediction
4.3. error analysis: print the ranked job titles from cosine comparison