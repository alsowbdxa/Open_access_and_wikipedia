import gc
import os
import pandas as pd
from tqdm import tqdm
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import random
from pandas.core.frame import DataFrame
import pickle
import sqlite3
tqdm.pandas()

data = pd.read_parquet('Wikipedia_citation_dataset.parquet') #find the dataset from: https://github.com/Harshdeep1996/cite-classifications-wiki

#merge the wikipedia dataset with the openalex data.
wiki_oa = data[['page_title','doi']]
wiki_oa['doi_updated'] = wiki_oa['doi'].progress_apply(lambda x:x.replace('\\n','').replace('\\t','').replace('\\\\','').replace('</a>','').strip())

open_alex_data = pd.read_parquet('openalex_data.parquet') # from get_data_from_OpenAlex.py
open_alex_data['doi_updated'] = open_alex_data['doi'].progress_apply(lambda x:x.split('https://doi.org/')[-1])

wiki_oa = pd.merge(wiki_oa,open_alex_data,on='doi_updated',how='left')#size is 1,705,085

# extract the needed variable
wiki_oa['is_oa'] = wiki_oa['open_access'].progress_apply(lambda x:x.get('is_oa', None))
wiki_oa['oa_status'] = wiki_oa['open_access'].progress_apply(lambda x:x.get('oa_status', None))
wiki_oa['journal'] = wiki_oa['host_venue'].progress_apply(lambda x:x.get('display_name',None))
wiki_oa['macro_concepts'] = wiki_oa['concepts'].progress_apply(lambda x:[i['id'] for i in x if i['level']==0])
wiki_oa['num_macro_concepts'] = wiki_oa['macro_concepts'].progress_apply(lambda x:len(x))
wiki_oa['num_references'] = wiki_oa['referenced_works'].progress_apply(lambda x:len(x))
wiki_oa['is_oa'].replace(True,1,inplace=True)
wiki_oa['is_oa'].replace(False,0,inplace=True)
wiki_oa['publication_year'] = wiki_oa['publication_date'].progress_apply(lambda x:x.split('-')[0])

##################################################################################################
# plot figure 1 in the paper.

sns.set_context("paper" )
#Figure 1.1: plot the distribution of oa status by citations
al=sns.color_palette(['#d3494e','#8B4513','#019529','#dbb40c','#9370DB'])
x = list(wiki_oa['oa_status'].value_counts(normalize=True).index)
label = list(wiki_oa['oa_status'].value_counts(normalize=True).values)
label = ["%.2f%%" % (i*100) for i in label]
y = list(wiki_oa['oa_status'].value_counts().values)

plt.figure(dpi=600)
plt.ticklabel_format(style='plain')
ax = sns.barplot(x=x,y=y,palette=al)
for i in range(5):
    ax.text(i,y[i]+10000,label[i],ha="center",fontsize=14)
ax.set_ylabel("Number of Citations", fontsize=14)
plt.yticks([i*200000 for i in range(7)],['0','200k','400k','600k','800k','1000k','1200k'],fontsize=14)
ax.tick_params(labelsize=14)
plt.tight_layout()

#Figure 1.2: plot the distribution of oa status by unique dois

al=sns.color_palette(['#d3494e','#8B4513','#019529','#dbb40c','#9370DB'])
x = list(wiki_oa.drop_duplicates(subset=['doi_y'])['oa_status'].value_counts(normalize=True).index)
label = list(wiki_oa.drop_duplicates(subset=['doi_y'])['oa_status'].value_counts(normalize=True).values)
label = ["%.2f%%" % (i*100) for i in label]
y = list(wiki_oa.drop_duplicates(subset=['doi_y'])['oa_status'].value_counts().values)

plt.figure(dpi=600)
plt.ticklabel_format(style='plain')
ax = sns.barplot(x=x,y=y,palette=al)
for i in range(5):
    ax.text(i,y[i]+10000,label[i],ha="center",fontsize=14)
ax.set_ylabel("Number of unique Citations", fontsize=14)
plt.yticks([i*200000 for i in range(5)],['0','200k','400k','600k','800k'],fontsize=14)
ax.tick_params(labelsize=14)
plt.tight_layout()
##################################################################################################

##################################################################################################
# plot figure 2
x = []
y = [[],[]]
for i in tqdm(wiki_oa.groupby('publication_year')):
    x.append(i[0])
    y[0].append(i[1]['is_oa'].sum()/len(i[1]))
    y[1].append(len(i[1])/1705085)

plt.figure(dpi=600)

plt.rcParams['figure.dpi'] = 600
fig, ax1 = plt.subplots()
ax1 = sns.lineplot(x=x[-41:-2],y=y[0][-41:-2],color='blue')
ax1.set_ylabel('Fraction of OA')
#set the interval as 5 years
labels = [x[-41:-1][i] if i%5==0 else '' for i in range(len(x[-41:-1]))][:-1]

ax2 = ax1.twinx()
ax2 = sns.lineplot(x=x[-41:-2],y=y[1][-41:-2],color = 'black')
ax2.set_ylabel('Fraction of citations')
plt.xticks(x[-41:-2], labels, rotation='vertical') 
fig.legend(['Fraction of OA by year','Fraction of citations by year'],bbox_to_anchor=(0.53, 0.97))
plt.tight_layout()
##################################################################################################

##################################################################################################
# figure 3
data_concepts_is_oa['total_num'] = data_concepts_is_oa.progress_apply(lambda x:x['is_oa']+x['not_oa'],axis=1)

x = list(data_concepts_is_oa.sort_values('total_num').index)[::-1]
y = data_concepts_is_oa.sort_values('total_num')['total_num'].to_list()[::-1]

pl = sns.color_palette("Blues_r",30)
plt.figure(dpi=600)
grid = plt.GridSpec(1, 7, wspace=0.2, hspace=0.5)
plt.subplot(grid[0,0:5])
ax1 = sns.barplot(y=data_concepts_is_oa.index,x=data_concepts_is_oa['is_oa_p']+data_concepts_is_oa['not_oa_p'],
                  color = "#f33030",order=x,alpha=0.6,label='not OA')

sns.barplot(y=data_concepts_is_oa.index,x = data_concepts_is_oa['is_oa_p'],
            color = "#8795e2",data=data_concepts_is_oa,order=x,alpha=1,label='OA')

legend_handles = [plt.Line2D([], [], color='#8795e2',linewidth=6),
                  plt.Line2D([], [], color='#f33030',alpha=0.6,linewidth=6)]
plt.legend(handles=legend_handles, labels=['OA','not OA'],bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
            mode="expand", borderaxespad=0, ncol=4,fontsize=9,frameon=False,
            )

ax1.set_xlabel('Percentage of citations', fontsize=12)
plt.axvline(0.391, color='black', linestyle='--')
plt.subplot(grid[0,5:7])
ax2 = sns.barplot(x=y, y=x, palette=pl)
ax2.set_yticklabels([])
ax2.set_xlabel('Number of citations', fontsize=12)
##################################################################################################

##################################################################################################
# plot figure 4
concept_id = ['17744445',
 '138885662',
 '162324750',
 '144133560',
 '15744967',
 '33923547',
 '71924100',
 '86803240',
 '41008148',
 '127313418',
 '185592680',
 '142362112',
 '144024400',
 '127413603',
 '205649164',
 '95457728',
 '192562407',
 '121332964',
 '39432304']

concept_name = ['Political science',
 'Philosophy',
 'Economics',
 'Business',
 'Psychology',
 'Mathematics',
 'Medicine',
 'Biology',
 'Computer science',
 'Geology',
 'Chemistry',
 'Art',
 'Sociology',
 'Engineering',
 'Geography',
 'History',
 'Materials science',
 'Physics',
 'Environmental science']

t = [i.split('.org/')[-1][1:] for i in t]#concepts id, see above 
dic_l = [Counter(dict(zip(t,[0 for i in t]))) for _ in range(5)]
oa_status = ['bronze', 'closed', 'gold','green','hybrid']
for i in tqdm(wiki_oa[['oa_status','macro_concepts']].groupby('oa_status')):
    for n in i[1]['macro_concepts'].to_list():
        n = [i.split('.org/')[-1][1:] for i in n]
        v = dict(zip(n,[1/len(n) for _ in n]))
        dic_l[oa_status.index(i[0])].update(v)
test = pd.DataFrame([i for i in dic_l])
test.index = oa_status
test.columns = concept_name
data_concepts_oa_status = test
data_concepts_oa_status = data_concepts_oa_status.T

data_concepts_oa_status = data_concepts_oa_status.T[x[::-1]].T

data_concepts_oa_status = data_concepts_oa_status[[ 'closed','bronze',  'green','gold', 'hybrid']]
plt.rcParams['figure.dpi'] = 600
data_concepts_oa_status.plot(kind='barh',stacked=True,color = ['#d3494e','#8B4513','#019529','#dbb40c','#9370DB'])
plt.xlabel('Number of Citations')
##################################################################################################

##################################################################################################
# plot figure 7 and figure 8

# figure 7
import seaborn.objects as so
journal_is_oa = wiki_oa[['journal','is_oa']]
l=journal_is_oa.groupby('journal').size()
l.sort_values(inplace=True)
top20_journal = list(l[-20:].index)[::-1]
t=[]
for i in tqdm(journal_is_oa.groupby('journal')):
    if i[0] in top20_journal:
        t.append([i[0],i[1]['is_oa'].sum(),len(i[1])])
        
t = pd.DataFrame(t)
t.columns=['journal','is_oa','num_of_citations']
t['not_oa'] = t.progress_apply(lambda x:x['num_of_citations']-x['is_oa'],axis=1)
t.drop(columns=['num_of_citations'],inplace=True)

#sort the journal
t.index = t['journal']
t = t.loc[top20_journal[::-1]]

plt.rcParams['figure.dpi'] = 600
t.plot(kind='barh',stacked=True)
plt.xlabel('Number of Citations')
plt.ylabel('Top20 Journals')


# figure 8
journal_oa_status = wiki_oa[['journal','oa_status']]
t=[]
for i in tqdm(journal_oa_status.groupby('journal')):
    if i[0] in top20_journal:
        dic = {'bronze':0, 'closed':0,'gold':0, 'green':0,'hybrid':0}
        v = i[1]['oa_status'].value_counts()
        dic.update(v)
        r = [i[0]]+list(dic.values())
        t.append(r)
data_top20_journal_oa_status = pd.DataFrame(t)
data_top20_journal_oa_status.columns=['journal','bronze', 'closed','gold', 'green','hybrid']
data_top20_journal_oa_status.index = data_top20_journal_oa_status['journal']
data_top20_journal_oa_status = data_top20_journal_oa_status.loc[top20_journal[::-1]]

# change the order
data_top20_journal_oa_status = data_top20_journal_oa_status[['journal',  'closed','bronze', 'green', 'gold', 'hybrid']]

plt.rcParams['figure.dpi'] = 600
data_top20_journal_oa_status.plot(kind='barh',stacked=True,color = ['#d3494e','#8B4513','#019529','#dbb40c','#9370DB'])
plt.xlabel('Number of Citations')
plt.ylabel('Top20 Journals')
##################################################################################################



##################################################################################################
# start to do regression
# firstly, get the stratified sample

# the below method is considering 3 aspects: journal, publication year and concept, for 2 aspects (only publication year and journal), just need to remove the related process with concept.

fc_sample = wiki_oa.drop_duplicates(subset=['doi_updated']).dropna(subset=['oa_status']).dropna(subset=['concepts'])
fc_sample = fc_sample[fc_sample['num_macro_concepts']==1]

#remove the published year earlier than 1900
fc_sample = fc_sample[fc_sample['publication_year']>='1900']
#remove the 2022 because there is no data on scimg of 2022
fc_sample = fc_sample[fc_sample['publication_year']!='2022']

fc_sample.dropna(subset=['journal'],inplace=True)
journal = fc_sample.groupby('publication_year')['journal'].unique()

#aggregate the journals before 1999
journal_before_1999 = [n for i in range(1900,2000) for n in journal[str(i)]]
journal_before_1999 = list(set(journal_before_1999))

# for special characters
frde_label = str.maketrans("éàèùâêîôûçüöä", "eaeuaeioucuoa")

scimg_data=pd.DataFrame()
# read the data from scimg, which you can download directly from: https://www.scimagojr.com/journalrank.php
for i in tqdm(list(journal.keys())):
    if i>'1999':
        scimg = pd.read_csv(r'scimagojr {}.csv'.format(i),sep=';', low_memory=False)
        scimg['year'] = i
        j = eval(str.translate(str(list(journal[i])),frde_label))
        scimg_data = pd.concat([scimg_data,scimg[scimg['Title'].isin(j)]])
   
scimg = pd.read_csv(r'scimagojr {}.csv'.format(1999),sep=';', low_memory=False)
scimg['year'] = '1999'
j = eval(str.translate(str(journal_before_1999),frde_label))
scimg_data = pd.concat([scimg_data,scimg[scimg['Title'].isin(j)]])


scimg_data = scimg_data[['Rank', 'Title', 'SJR', 'SJR Best Quartile',
       'H index', 'Total Refs.', 'Ref. / Doc.', 'Country', 'Region',
       'Publisher', 'Coverage', 'Categories', 'year']]
#remove journals that have less than 20 citations
scimg_data = scimg_data[scimg_data['Total Refs.']>=20]

#drop the duplicates
scimg_data.drop_duplicates(subset=['Title','year'],inplace=True)

#match it back to fc_data
fc_sample['year'] = fc_sample['publication_year'].progress_apply(lambda x: x if x>='1999' else '1999')
fc_sample['journal_title'] = fc_sample['journal'].progress_apply(lambda x: str.translate(x,frde_label))
fc_sample = pd.merge(fc_sample,scimg_data,left_on=['journal_title','year'],right_on=['Title','year'],how='left')

# remove the nan in the key variables we need
fc_data_wiki = fc_sample.dropna(subset=['cited_by_count','publication_date','macro_concepts','SJR','oa_status','is_oa','num_references'])

#calculate the article age(how many month)
def cal_month(x):
    x = x.split('-')
    return (2022-int(x[0]))*12+12-int(x[1])

fc_data_wiki['article_age'] = fc_data_wiki['publication_date'].progress_apply(lambda x:
            cal_month(x))

# save the fc_data_wiki
fc_data_wiki.to_parquet(r'fc_data_wiki.parquet')

# then use it to extract the stratified sample from openalex

# stratified by these three variables:
# – Journal
# – Year of publication
# – Field of concepts

fc_data_wiki['concepts_id'] = fc_data_wiki['macro_concepts'].progress_apply(lambda x: x[0].split('/')[-1])

s_group = fc_data_wiki.groupby(['journal','publication_year','concepts_id'])

# then you need use the method in get_data_from_OpenAlex.py to extract data from openalex to generate the stratified sample.
# after getting the data from openalex, we could start to stratify
# the data from openalex could be saved as a dictionary, the key is the strata, and the value is the corresponding article data from openalex. This file saved as "openalex_dict.pkl"

# this dictionary store the data from openalex
with open('openalex_dict.pkl', 'rb') as f:
    openalex_dict = pickle.load(f)

###############################################################################
# start to stratify
stra_qc = [] # store the extracted strata
error = []
stratified_academia_data = [[],[],[],[],[]]#save the negative dataset(article not in wikipedia)
article_in_wiki = pd.DataFrame()#save the article in wikipedia)

for i in tqdm(s_group):
    concept_id = i[0][-1]
    year = i[0][1]
    journal=i[0][0]
    url = 'https://api.openalex.org/works?filter=concepts.id:{},publication_year:{},type:journal-article,host_venue.display_name:{}&page=1&per-page=200'.format(concept_id,year,journal)
    if i[0] in stra_qc:
        continue
    if i[0] in error:
        continue
    res = openalex_dict[url]
    doi_in_wiki = i[1]['doi_updated'].to_list()
    doi_not_in_wiki = [i for i in res if i['doi'] not in doi_in_wiki]#remove the doi in our wiki dataset
    try:
        if res =='':
            stra_qc.append(i[0])
            error.append(i[0])
            continue
        else:
            total_count = len(doi_not_in_wiki)
    except:
        stra_qc.append(i[0])
        error.append(i[0])
        continue
    #If the total number of queries on the first page is less than the number of articles contained in that strata and there is no second page, sampling is not possible, and the strata should be discarded.
    if total_count<len(i[1]) and total_count<200:
        stra_qc.append(i[0])
        continue
    #if not, start to stratified sample
    ###################################
    #if the number of articles in first page greater than strata, random choice
    
    if total_count>=len(i[1]):
        for loop in range(5):
            neg_t = random.choices(doi_not_in_wiki,k=len(i[1]))
            stratified_academia_data[loop].extend(neg_t)
    #if first page has 200 results but still less than the strata, extract more
    elif total_count==200 and total_count<len(i[1]):
        neg_t = res
        page=2
        print('num of strate/results:{}/{}, start to extract page 2'.format(len(i[1]),total_count))
        while len(neg_t)<len(i[1])*3:
            url = 'https://api.openalex.org/works?filter=concepts.id:{},publication_year:{},type:journal-article,host_venue.display_name:{}&page={}&per-page=200'.format(concept_id,year,journal,page)
            res = json.loads(requests.get(url,headers=headers,timeout=30).text)
            d = [{k:v for k,v in dic_t.items() if k in ['cited_by_count','title',
                    'doi','host_venue','open_access','publication_date','referenced_works']} for dic_t in res['results']]
            for n in d:
                n['num_reference'] = len(n['referenced_works'])
                t = n.pop('referenced_works')
            neg_t.extend(d)
            page+=1
            if page>10:
                break
        if page>10:
            stra_qc.append(i[0])
            error.append(i[0])
            continue
        else:
            doi_not_in_wiki = [i for i in neg_t if i['doi'] not in doi_in_wiki]#remove the doi in our wiki dataset
            for loop in range(5): #randomly get 5 samples.
                stratified_academia_data[loop].extend(random.choices(doi_not_in_wiki,k=len(i[1])))
    else:
        stra_qc.append(i[0])
        error.append(i[0])
        continue
    article_in_wiki=pd.concat([article_in_wiki,i[1]],ignore_index=True)
    stra_qc.append(i[0])
###############################################################################

article_in_wiki['is_wiki'] = 1

# create sjr to match with sample data
sjr = article_in_wiki[['journal_title','year','SJR','SJR Best Quartile','H index',
                       'Total Refs.', 'Ref. / Doc.', 'Country', 'Region', 'Publisher',
                       'Coverage', 'Categories']]
sjr.drop_duplicates(subset=['journal_title','year'],inplace=True)


sample_1 = pd.DataFrame(stratified_academia_data[0]
sample_1['macro_concepts'] = article_in_wiki['macro_concepts']
sample_1['is_wiki'] = 0
#unstack the data
sample_1['publication_year'] = sample_1['publication_date'].progress_apply(lambda x:x.split('-')[0])
# "year" is from 1999 to 2022, only for match with sjr 
sample_1['year'] = sample_1['publication_year'].progress_apply(lambda x: x if x>='1999' else '1999')

sample_1['is_oa'] = sample_1['open_access'].progress_apply(lambda x:x['is_oa'])
sample_1['oa_status'] = sample_1['open_access'].progress_apply(lambda x:x['oa_status'])
# calculate the article age
#use the same end date with article in wikipedia
sample_1['article_age'] = sample_1['publication_date'].progress_apply(lambda x:
            cal_month(x))
    
sample_1['journal'] = sample_1['host_venue'].progress_apply(lambda x:x['display_name'])
sample_1['journal_title'] = sample_1['journal'].progress_apply(lambda x: str.translate(x,frde_label))

sample_1 = pd.merge(sample_1,sjr,on=['journal_title','year'],how='left')

#change variable name num_reference to num_references in sample to match the 
#data in article_in_wiki
sample_1.rename(columns={'num_reference':'num_references'},inplace=True)

# the same process can be done for the other samples

###############################################################################
# combine stratified_sample and article_in_wiki
# with num_reference
t1 = article_in_wiki[['doi_updated','cited_by_count','num_references',
                      'article_age','macro_concepts','SJR', 'SJR Best Quartile',
                      'H index','is_oa', 'oa_status','is_wiki']]

t1.columns=['doi','cited_by_count','num_references','article_age',
            'macro_concepts','SJR','SJR Best Quartile', 'H index','is_oa',
            'oa_status','is_wiki']

# with num_reference
t2 = sample_1[['doi','cited_by_count','num_references','article_age',
               'macro_concepts','SJR','SJR Best Quartile', 'H index','is_oa',
               'oa_status','is_wiki']]
###############################################
# combine t1 and t2
reg_sample_1 = pd.concat([t1,t2])#dataset to do regression analysis
# process to make it ready for regression

reg_sample_1['ln1p_num_references'] = reg_sample_1['num_references'].progress_apply(lambda x: np.log(1+x))
reg_sample_1['ln1p_times_cited'] = reg_sample_1['cited_by_count'].progress_apply(lambda x: np.log(1+x))
reg_sample_1['ln_article_age'] = reg_sample_1['article_age'].progress_apply(lambda x: np.log(x))
reg_sample_1['ln1p_SJR'] = reg_sample_1['SJR'].progress_apply(lambda x: np.log(1+float(x.replace(',','.'))))
reg_sample_1['is_oa'].replace(True,1,inplace=True)
reg_sample_1['is_oa'].replace(False,0,inplace=True)
# use concept_id and concept_name above
reg_sample_1['concept'] = reg_sample_1['macro_concepts'].progress_apply(lambda x: concept_dic[x[0].split('https://openalex.org/C')[-1]])
                      
###############################################
                        
# use logistic regression
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col                        
                        
formula_1 = "is_wiki~ln1p_article_age+ln1p_SJR+is_oa"
model_logit = smf.logit(formula=formula_1,data=reg_sample_1)
#fit the model
res_logit = model_logit.fit()
print(res_logit.summary())
                        

formula_2 = "is_wiki~ln1p_article_age+ln1p_SJR+is_oa+ln1p_times_cited"
model_logit = smf.logit(formula=formula_2,data=reg_sample_1)
#fit the model
res_logit = model_logit.fit()
print(res_logit.summary())


                        
