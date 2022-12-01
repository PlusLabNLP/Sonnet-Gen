#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
tok = BartTokenizer.from_pretrained("facebook/bart-base")
import yake
import json


# In[11]:


kw_extractor = yake.KeywordExtractor(n = 1, top =3)


# In[12]:


def has_numbers(inputString):
     return any(char.isdigit() for char in inputString)


# In[13]:


path = '../data/news_summary/'
filename ='news_summary_more.csv'
news, prompts = [], []
df = pd.read_csv(path+filename)


# In[14]:


df


# In[15]:


headlines = df['headlines']
news = df['text']


# In[20]:


for i in news[:10]:
    print(i)


# In[24]:


tokens = []
sentences = []
paragraphs = []
count = 0
filtered_stories = []
filtered_prompts = []
for s,p in zip(news, headlines):
    if has_numbers(p) or pd.isna(s) or '@'in s or '#'in s:
        continue
    try:
        s = s.strip('\n').replace('?s',"'s").replace('?','').replace('..','.')
        s = s.replace('..','.').replace('\n', '').replace("\'","'")
        s = s.replace('..','.')
        token_l = len(s.split())
        batch = tok(s, return_tensors='pt')
        sent_l = len(s.split('.'))
        sents = s.split('.')
        flag = True
        #print(len(batch['input_ids'][0]) /token_l)
        if 8<=sent_l<=50:
        #if len(batch['input_ids'][0]) < 1.5*token_l and 8<=sent_l<=50:
            count += 1
            tokens.append(token_l)
            sentences.append(sent_l)
            filtered_stories.append(s)
            filtered_prompts.append(p)
    except:
        continue

print(count)    
print('average tokens of story:', np.mean(tokens))
print('average sentences of story:', np.mean(sentences))
print('average tokens per sentence:', np.mean(tokens)/np.mean(sentences))


# In[8]:


len(filtered_stories), len(filtered_prompts)


# In[9]:


all_story = []
for s, p in zip(filtered_stories, filtered_prompts):
    p = p.replace('\n', '')
    story = {}
    story['Theme'] = p
    sents = s.split('.')
    story_keywords = []
    informative_lines = []
    sentiments = []
    for sent in sents:
        kws = kw_extractor.extract_keywords(sent)
        keywords = [kw[0] for kw in kws]
        if len(keywords) >= 2:
            informative_lines.append(sent)
            story_keywords.append(keywords)
    story['sentiments'] = sentiments
    story['keywords'] = story_keywords
    story['sentences'] = informative_lines
    all_story.append(story)
    if len(all_story) % 500 == 1:
        print(len(all_story))
        json.dump(all_story, open('all_news_short_theme''.json','w'))


# In[ ]:




