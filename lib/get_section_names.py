import re 
"""
keys = []
for i in range(len(JD_ls_raw)):
    keys+=[k for k,v in JD_ls_raw[i].items()]
#keys = list(set(keys))
count = {}
for i in range(len(keys)):
    try:
        count[keys[i]]+=1
    except:
        count[keys[i]]=1

import operator 
count = sorted(count.items(), key=operator.itemgetter(1),reverse=True)
count =[(k,v) for k,v in count if v>1 and len(k.split())<4]

for k in count:
    pat = 'qualifi'
    pat = 'educa'
    pat = 'experie'
    pat = 'salar'
    pat = 'remuner'
    pat = 'skills'
    pat = 'descrip'
    pat = 'title'
    pat = 'dura'
    pat = 'loca'
    pat = 'term'
    pat = 'knowledge'
    pat = 'about'
    pat = 'resp'
    if re.search(pat,k[0].lower())!=None:
        print("\'"+k[0]+"\',")
"""
section_names = {}
section_names['description'] = \
['JOB DESCRIPTION',
'DETAIL DESCRIPTION',
'DESCRIPTION',
'ArmeniaJOB DESCRIPTION',
'Job Description',
'NKRJOB DESCRIPTION',]

section_names['about'] = \
['ABOUT COMPANY',
'ABOUT',
'About Project',
#'About the trainers',
]

section_names['responsibility'] = \
['JOB RESPONSIBILITIES',
'Other responsibilities',
'Other Responsibilities',
'responsibilities',
'General Responsibilities',
'Major Responsibilities',
'RESPONSIBILITIES',
'responsibilities include',
'Specific Responsibilities',
#'Secondary Responsibilities',
'Responsibilities include',
'Corresponding professional education',
'Key Responsibilities',
#'Incident  Emergency Response',
#'of responsibility',
]
section_names['qualification_required'] = \
['REQUIRED QUALIFICATIONS',
'Qualifications',
'qualifications',
'Qualifications and skills',
'of qualifications to',
'Required Qualifications',
'Minimum Qualifications',]

section_names['qualification_desired'] = \
['Desired qualifications',
'Desired Qualifications',
'PREFERRED QUALIFICATIONS',
'DESIRED QUALIFICATIONS',
'Preferred Qualifications',
'DESIRABLE QUALIFICATIONS',
'Preferred qualifications',
'Desirable Qualifications',

'Additional Qualifications',
'Other qualifications',
'Other Qualifications',
#'teaching qualifications to',
]

section_names['education'] = \
['EDUCATIONAL LEVEL',
'Minimum Academic Qualifications',
'Education',
'EDUCATION TYPE',
'education to',
'professional education to',
'education',
'International Education',]

section_names['experience']=\
['Experience',
'and experience to', 
'General experience',
'experience to', 
'relevant experience to',
'Experience with',
'Specific professional experience',
'WORK EXPERIENCE',
'Professional experience',
'General professional experience',
'Desired Experience',
'Preferred work experience',
'experience',
'Prior Work Experience',
'general experience',
'Desirable Experience',
'Work experience',
'Software experience requirements',
'and experience',]

section_names['salary']=\
['REMUNERATION',
'expected salaries to',
'expected salary to',]

section_names['skill'] = \
['Desired Skills',
'Desired skills',
'Skills',
'Professional skills',
'DESIRED SKILLS',
'Additional skills',
# 'Qualifications and skills', #this should go to qualifications
'Preferred Skills',
'Computer Skills',
'Capacity and Skills',
'Language skills',
'Desirable Skills',
'Required skills',
'Skills and competencies',
'People Skills',
'Capacity and skills',
'Desirable skills',
'Technical Skills',
'Personal skills',
'Project management skills',
'Skills and Abilities',
'Professional Skills',
'Computer skills',
'Technical skills',
'Personal Skills',
'Preferred skills',
'SKILLS',
'Knowledge of',
'DOMAIN KNOWLEDGE',
'Excellent knowledge of',
'Knowledge',
'Knowledge and skills',
]

section_names['organization']=\
['ORGANIZATION',]

section_names['title']=\
['TITLE',
'JOB TITLE',]
section_names['duration']=\
['DURATION',
'PROJECT DURATION',
'POSITION DURATION',
'Duration',
'Course duration',]

section_names['location'] = \
['LOCATION',
'POSITION LOCATION',
'or located at',
'located at',
'local governance by',]

import pickle
section_names_path = './../tmp/section_names.pickle'
with open(section_names_path, 'wb') as handle:
    pickle.dump(section_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
