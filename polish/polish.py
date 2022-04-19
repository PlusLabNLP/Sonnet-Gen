#!/usr/bin/env python
# coding: utf-8

# In[1]:


#example
four_seasons_story_line = [
['snow', 'falling', 'future'],
['winter', 'is', 'coming'],
['gather', 'honest', 'humor'],
['spring', 'happy', 'blooming'],
['air', 'heat', 'warm'],
['little', 'birds', 'may'],
['flowers', 'leaves', 'storm'],
['summer','moon', 'day'],
['blue', 'sky', 'clouds'],
['sudden', 'rain', 'thunder'],
['Summer', 'fill', 'crowds'],
['Spring', 'no', 'wonder'],
['seasons','years', 'keep'],
['future', 'months', 'reap']]


# In[23]:


import random


# ### Imagery for nouns

# In[3]:


import spacy
nlp = spacy.load('en_core_web_sm')


# In[ ]:


import os
import sys
import argparse
import torch

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive


# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
cfg.device = "cpu"


# In[ ]:


#load model
model_file =  'pretrained_models/reverse_comet_1e-05_adam_32_20000.pickle'
opt, state_dict = interactive.load_model_file(model_file)
data_loader, text_encoder = interactive.load_data("conceptnet", opt)

n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
n_vocab = len(text_encoder.encoder) + n_ctx

model = interactive.make_model(opt,  40543, 29, state_dict)


# In[ ]:



def getloss(input_e1, input_e2, relation, prnt = False):
    if relation not in data.conceptnet_data.conceptnet_relations:
        if relation == "common":
            relation = common_rels
        else:
            relation = "all"
    outputs = interactive.evaluate_conceptnet_sequence(
        input_e1, model, data_loader, text_encoder, relation, input_e2)

    for key, value in outputs.items():
        if prnt:
            print("{} \t {} {} {} \t\t norm: {:.4f} \t".format(
                input_e1, key, rel_formatting[key], input_e2, value['normalized_loss']))
        return round(value['normalized_loss'],4)


# In[ ]:


def getPred(input_event, relation, prnt = True, sampling_algorithm = 'beam-2'):
    sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)
    outputs = interactive.get_conceptnet_sequence(input_event, model, sampler, data_loader, text_encoder, relation, prnt)
    return outputs


# In[48]:


#randomly sample at most N=5 nouns, not from the same line


# In[24]:


N = 5
M = 2
if __name__ == "__main__":
    location_dict = {}
    for i, keywords in enumerate(four_seasons_story_line):
        w1, w2, _ = keywords
        ent = nlp(w1)[0]
        if ent.pos_ == 'NOUN':
            location_dict[str(ent)] = [i,0]
            continue
        ent = nlp(w2)[0]
        if ent.pos_ == 'NOUN':
            location_dict[str(ent)] = [i,1]
    samples = random.sample(location_dict.keys(),N)
    relations = ['SymbolOf']
    score_dict = {}
    replace_dict = {}
    polished_lines = []

    for ent in samples:
        result = getPred(ent, relation=relations, sampling_algorithm = 'beam-5', prnt = False)
        result = result[relations[0]]['beams'][0]
        score_dict[ent] = getloss(ent, result, 'SymbolOf', prnt = False)
        replace_dict[end] = result

    selected = sorted(score_dict.items(), key=lambda item: item[1])[:M]
    for ent in selected:
        ent = ent[0]
        location = location_dict[ent]
        polished_lines.append(location[0])
        four_seasons_story_line[location[0]][location[1]] = replace_dict[ent]


# ### Simile for adjs

# In[ ]:


from fairseq.models.bart import BARTModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import numpy as np
import pickle


# In[ ]:


datadir = 'simile'
cpdir = datadir+'/'
bart = BARTModel.from_pretrained(cpdir,checkpoint_file='checkpoint-simile/checkpoint_best.pt',data_name_or_path=datadir)

bart.cuda()
bart.eval()


maxb = 30
minb = 2
t = 0.7


# In[ ]:


def simile_vehicle(inp):
        last_word = inp.split()[-1]
        inp = inp.replace(' so ',' ')
        slines = [inp]
        answer = ''
        
        while True:
            hypotheses_batch = bart.sample(slines, sampling=True, sampling_topk=5, temperature=t, lenpen=2.0, max_len_b=maxb, min_len=minb, no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                    answer = hypothesis.replace('\n','')
            vehicle = answer.strip().split(' like ')[1].replace('<EOT>','')
            if check_meter(vehicle):
                return vehicle


# In[62]:


if __name__ == "__main__":
    location_dict = {}
    for i, keywords in enumerate(four_seasons_story_line):
        if i not in polished_lines:
            w1, w2, _ = keywords
            ent = nlp(w1)[0]
            if ent.pos_ == 'ADJ':
                location_dict[str(ent)] = [i,0]
                continue
            ent = nlp(w2)[0]
            if ent.pos_ == 'ADJ':
                location_dict[str(ent)] = [i,1]
                
    samples = random.sample(location_dict.keys(),M)
    for ent in samples:
        vehicle = simile_vehicle(ent)
        location = location_dict[ent]
        polished_lines.append(location[0])
        four_seasons_story_line[location[0]][location[1]] = vehicle

