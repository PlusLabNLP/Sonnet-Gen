
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPTNeoModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import pipeline
import requests
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import streamlit as st
from bs4 import BeautifulSoup

from torch import cuda
device = 'cuda:0' if cuda.is_available() else 'cpu'

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("FigoMe/sonnet_keyword_gen")
model = AutoModelForSeq2SeqLM.from_pretrained("FigoMe/sonnet_keyword_gen")
model = model.to(device)


nlp = pipeline("fill-mask", model=model, tokenizer = tokenizer, device=0)

def generate(user_input):
    prompt = "Generate keywords for the title: "
    placeholder = ". Keywords 1: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 2: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 3: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 4: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 5: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 6: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 7: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 8: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 9: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 10: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 11: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 12: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 13: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 14: ['<MASK>', '<MASK>', '<MASK>'] </s>"
    bart_input = prompt + user_input + placeholder
    ids = tokenizer(bart_input, return_tensors="pt").input_ids.to(device)
    generated_ids = model.generate(
                  input_ids = ids,
                  max_length=512, 
                  min_length = 200,
                  num_beams = 4,
                  no_repeat_ngram_size = 5,
                  #topp = 0.9,
                  #do_sample=True,
                  repetition_penalty=5.8, 
                  length_penalty=1, 
                  early_stopping=True
                  )
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids] 
    return str(preds).replace('.','\n')

def get_rhyme(word, N):
  header = {'User-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10) AppleWebKit/600.1.25 (KHTML, like Gecko) Version/8.0 Safari/600.1.25'}
  url = "https://www.wordhippo.com/what-is/words-that-rhyme-with/" + word + ".html"
  page = requests.get(url, headers=header)
  print(page)
  soup = BeautifulSoup(page.content, 'html.parser')
  rhymes = [a.getText() for a in soup.select_one('div.relatedwords').find_all("a")]
  return rhymes[:N]

def rhyme_gen(sentence, add_radio):

    if add_radio == 'Petrarchan':
        initial_rhyming_lines = [0,1,8, 9, 10]
        countin_rhyming_lines = [11, 12, 13]
        #countin_rhyming_lines2 = [3,2,5,4,7,6]
        st.write('Sidebar works')
    else :
        initial_rhyming_lines = [0,1,4,5,8,9,12]
        countin_rhyming_lines = [2,3,6,7,10,11,13]
        st.write('Sidebar Works shakesy')

    for i in range(7):
        word = sentence.split(' Keywords')[initial_rhyming_lines[i]].split("'")[5]
        candidates = get_rhyme(word, N=30)
        replace_word = sentence.split(' Keywords')[countin_rhyming_lines[i]].split("'")[5]
        mask_input =  sentence.replace(replace_word,nlp.tokenizer.mask_token)
        result = nlp(mask_input, top_k = 5000)
        tokens = [res['token_str'] for res in result]
        found = False
        for t in tokens:
            if t in candidates:
                st.write('generated rhyme word:', t)
                found = True
                break
        
        if not found:
            candidates = [' '+c for c in candidates]
            result = nlp(mask_input, targets = candidates)
            tokens = [res['token_str'] for res in result]
            for t in tokens:
                if t in candidates:
                    print('generated rhyme word:', t)
                    found = True
                    break
            if not found : #if not found in masking outputs use random token to replace word, change 2 to randint
                st.write('generated rhyme word:', candidates[2])
                t = candidates[2]

        sentence = sentence.replace(replace_word,t)
    
    return sentence
        

st.title("Zero-shot Sonnet Generation with Discourse-level Planning and Aesthetics Features")
write_here = "A blue lake"
title = st.text_input("Enter a title to generate the sonnet. Please also choose the type of sonnet from the sidebar", write_here)
keyword = generate(title)
# keyword_clean = keyword.replace('[', ' ')
# keyword_clean = keyword_clean.replace(']', ' ')

with st.sidebar:
    st.header('Zest : A Zero-Shot Sonnet Generation Model')
    st.write('Zest does not require training on any poetic data. It consists of four components: content planning, rhyme pairing, polishing for aesthetics, and final decoding. The first three steps \
    provide salient points for the sketch of a sonnet.The last step is responsible for “translating” the sketch into well-formed sonnets.')
    add_radio = st.radio(
        "Choose a type of sonnet",
        ("Petrarchan", "Shakespearean")
    )

if "Generate Keywords" not in st.session_state:
    st.session_state["Generate Keywords"] = False

if "Generate Rhyme Words" not in st.session_state:
    st.session_state["Generate Rhyme Words"] = False

if "Generate Sonnet" not in st.session_state:
    st.session_state["Generate Sonnet"] = False

st.subheader('Keyword Generation')
if st.button("Generate Keywords"):
    #st.text_area('generated kewords',value=generate(review), key='keyword1')
    st.session_state["Generate Keywords"] = not st.session_state["Generate Keywords"]
    
if st.session_state["Generate Keywords"] :  
    edited_keyword = st.text_area('For each piece of news or stories, we train a title-tokeywords framework that predicts the outline.\
     To this end, we first extract three most salient words per line using the RAKE (Rose et al., 2010) algorithm, which is a domain-independent keyword extraction technique',value=keyword)
    if st.button('Generate Rhyme Words'):
        st.session_state["Generate Rhyme Words"] = not st.session_state["Generate Rhyme Words"]

if st.session_state["Generate Keywords"] and st.session_state["Generate Rhyme Words"]:
    st.text_area('The generate rhyme words for the given keywords', value = rhyme_gen(edited_keyword, add_radio))
    if st.button('generate sonnet'):
        st.write('tea')
        

