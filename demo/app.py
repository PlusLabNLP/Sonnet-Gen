
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPTNeoModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import pipeline
import pronouncing
import requests
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import streamlit as st
from bs4 import BeautifulSoup
import random

import spacy
pos = spacy.load('en_core_web_sm')

from torch import cuda
device = 'cuda:2' if cuda.is_available() else 'cpu'

import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
kw_tokenizer = AutoTokenizer.from_pretrained("FigoMe/sonnet_keyword_gen")
kw_model = AutoModelForSeq2SeqLM.from_pretrained("FigoMe/sonnet_keyword_gen")
kw_model = kw_model.to(device)


nlp = pipeline("fill-mask", model=kw_model, tokenizer = kw_tokenizer, device=device)


torch.manual_seed(42)
sonnet_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", 
                                          bos_token="<|startoftext|>",
                            eos_token="<|endoftext|>",
                            pad_token="<|pad|>")

# # Download the pre-trained GPT-Neo model and transfer it to the GPU
# sonnet_model = GPTNeoForCausalLM.from_pretrained("FigoMe/news-gpt-neo-1.3B-keywords-line-by-line-reverse").cuda()
# # Resize the token embeddings because we've just added 3 new tokens 
# sonnet_model.resize_token_embeddings(len(sonnet_tokenizer))
# sonnet_model = sonnet_model.to(device)


##Global constants
single_character_word = ['i','a']
forbidden_words = ['dona','er','ira','ia',"'s","'m","hmm","mm"]






def get_stress(phone):
    stress = []
    for s in phone.split():
        if s[-1].isdigit():
            if s[-1] == '2':
                stress.append(0)
            else:
                stress.append(int(s[-1]))
    return stress

def alternating(stress):
    #Check if the stress and unstress are alternating
    check1 = len(set(stress[::2])) <= 1 and (len(set(stress[1::2])) <= 1)
    check2 = len(set(stress)) == 2 if len(stress) >=2 else True
    return (check1 and check2)

def get_phones(rhyme_word):
    phone = pronouncing.phones_for_word(rhyme_word)[0]
    stress = get_stress(phone)
    p_state = stress[0]
    n_syllables = len(stress)
    return p_state, n_syllables


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
    return_index = False
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        indices_keep = logits >= torch.topk(logits, top_k)[0][..., -1, None]
        indices_keep = indices_keep[0].tolist()
        indices_keep = [i for i,x in enumerate(indices_keep) if x == True]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    if return_index == True:
        return logits, indices_keep
    return logits


def reverse_order(line):
    line = line.replace(', ', ' , ')
    words = line.split()
    return ' '.join(reversed(words)).replace(' , ', ', ')

loose_list = ['that','is','of','the','it','a','as','with','like','go','to','on','in','at','are','and']
def check_either_stress(stress, source_word, loose = True):
    if loose and source_word in loose_list:
        return True
    if len(stress) == 1 and len(pronouncing.phones_for_word(source_word))>1:
                    phone0 = pronouncing.phones_for_word(source_word)[0]
                    phone1 = pronouncing.phones_for_word(source_word)[1]
                    stress0 = [int(s[-1]) for s in phone0.split() if s[-1].isdigit()]
                    stress1 = [int(s[-1]) for s in phone1.split() if s[-1].isdigit()]
                    if stress0+stress1 ==1 and stress0*stress1 == 0:
                        return True

    return False


def regularBeamSearch(prompts):
	'''
	Beam search that considers the coherence by adding a new variable: previously_generated_lines
	'''
	BeamScorer = {}
	for sentence in prompts:
		loss = score_gpt2(sentence)
		BeamScorer[sentence] = [loss]
	answers = sorted(BeamScorer.items(), key=lambda x: x[1], reverse=False)
	new_prompts = [ans[0] for ans in answers]
	return new_prompts

def score_gpt2(sentence, normalize = True):
	'''
	Score a single sentence using the vanilla gpt2 model finetuned on lyrics
	The default setting is to normalize because we won't face the issue mentioned in function "score".
	'''
	tokens_tensor = sonnet_tokenizer.encode(sentence, add_special_tokens=False, return_tensors="pt")[0].to(device)
	with torch.no_grad():
		loss = sonnet_model(tokens_tensor, labels=tokens_tensor)[0]
	if normalize:
		return loss/len(tokens_tensor)
	else:
		return loss

def myBeamSearch(prompts, all_states, all_n_sys, all_keywords, beam_size = 5):
    BeamScorer = {}
    return_seq, return_stt, return_sys, return_key = [], [], [], []
    for sentence, p_state, n_sys, keywords in zip(prompts, all_states, all_n_sys, all_keywords):
        loss = score(sentence)
        BeamScorer[sentence] = [loss, p_state, n_sys, keywords]
    answers = sorted(BeamScorer.items(), key=lambda x: x[1], reverse=True)
    new_prompts = [ans[0] for ans in answers]
    new_p_states = [ans[1][1] for ans in answers]
    new_n_sys = [ans[1][2] for ans in answers]
    new_keywords = [ans[1][3] for ans in answers]
    l = len(new_prompts)
    if l > beam_size:
        return_seq += new_prompts[0:beam_size]
        return_stt += new_p_states[0:beam_size]
        return_sys += new_n_sys[0:beam_size]
        return_key += new_keywords[0:beam_size]
    else:
        return_seq +=new_prompts
        return_stt += new_p_states
        return_sys += new_n_sys
        return_key += new_keywords
    return return_seq,return_stt, return_sys, return_key

def generate_next_word(input_ids1, temperature = 0.85, topk = 100, n_sample=10, device = device):
    current_word = 0
    original = sonnet_tokenizer.decode(input_ids1[0])
    for _ in range(1):
        outputs1 = sonnet_model(input_ids1)
        #print(outputs1)
        next_token_logits1 = outputs1[0][:, -1, :]
        next_token_logits1 = top_k_top_p_filtering(next_token_logits1, top_k=topk)
        logit_zeros = torch.zeros(len(next_token_logits1)).to(device)
        #logit_zeros = torch.zeros(len(next_token_logits1), device=device)

        next_token_logits = next_token_logits1 * temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=n_sample).squeeze(1)
        #unfinished_sents = torch.ones(1, dtype=torch.long, device=device)
        unfinished_sents = torch.ones(1, dtype=torch.long).to(device)
        tokens_to_add = next_tokens * unfinished_sents + sonnet_tokenizer.pad_token_id * (1 - unfinished_sents)

        temp = []
        for i in range(len(input_ids1)):
            temp +=[torch.cat([input_ids1[i].reshape(1,-1), token_to_add.reshape(1,-1)], dim=-1) for token_to_add in tokens_to_add[i]]
        input_ids1 = torch.stack(temp).view(len(temp),-1)
        # decode the generated token ids to natural words
        results = []
        input_ids1_l = []
        for input_id1 in input_ids1:
            gen = sonnet_tokenizer.decode(input_id1).replace(original,'').strip(' ')
            if len(gen.split()) >0:
                gen = gen.split()[0]
                gen = gen.lower()
                if gen not in results:
                    results.append(gen)
        return results

def score(sentence, normalize = True):
	'''
	Score a single sentence using the plan-to-lyrics model.
	The recommended setting is to NOT normalize, because the input sentence is very long: it contains the title, planed keywords, and previously generated lines. 
	In addition, the candidate sentences contain the same prefix (i.e., the title, planed keywords, and previously generated lines) and only differ in the currently generated line.
	Normaling means dividing the loss by a large factor which may result in similarity accross different candidate sentences.
	'''
	tokens_tensor = sonnet_tokenizer.encode(sentence, add_special_tokens=False, return_tensors="pt")[0].to(device)
	with torch.no_grad():
		loss = sonnet_model(tokens_tensor, labels=tokens_tensor)[0]
	if normalize:
		return loss/len(tokens_tensor)
	else:
		return loss


def get_valid_samples(prompt, p_state, n_syllables, keywords, n_sample=30, n_cands=5):
    #if n_syllables == 10 or n_syllables==11:
    if n_syllables == 10:
        return [prompt], [p_state], [n_syllables], [keywords]
    elif n_syllables > 10:
        return [], [], [],[]
    states = []
    all_n_syl = []
    
    prompts = []
    all_keywords= [] 
    #insert the keyword whenever possible
    for source_word in keywords:
        phone = pronouncing.phones_for_word(source_word)[0]
        stress = get_stress(phone)
        if not alternating(stress):
            continue

        #if the word is single syllable and can be either stressed or unstressed, flag = True
        flag = check_either_stress(stress, source_word)

        if stress[-1] == 1- p_state or flag:
            #print(source_word)
            states.append(stress[0])
            all_n_syl.append(n_syllables+len(stress))
            prompts.append(prompt+ ' ' + source_word )
            copy = keywords.copy()
            copy.remove(source_word)
            all_keywords.append(copy)    
    
    #The normal process of decoding
    input_ids = sonnet_tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    tokens = generate_next_word(input_ids, n_sample=n_sample)
    #print(tokens)
    for token in tokens:
        token = token.lower()
        if (len(token) == 1 and token not in single_character_word) or token in forbidden_words:
            continue
        if token not in prompt:
            try:
                phone = pronouncing.phones_for_word(token)[0]
                stress = get_stress(phone)
            except:
                continue
            if (not alternating(stress)) or (len(stress)==0):
                continue

            #if the word is single syllable and can be either stressed or unstressed, flag = True
            flag = check_either_stress(stress, token)

            if (stress[-1] == 1- p_state) or flag:
                tokens.append(token)
                if stress[-1] == 1- p_state:
                    states.append(stress[0])
                elif flag:
                    states.append(1- p_state)
                all_n_syl.append(n_syllables+len(stress))
                prompts.append(prompt+ ' ' + token )
                all_keywords.append(keywords)
                if len(prompts)>= n_cands:
                    return prompts, states, all_n_syl, all_keywords
    return prompts, states, all_n_syl, all_keywords

def generate_kws(user_input):
    prompt = "Generate keywords for the title: "
    placeholder = ". Keywords 1: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 2: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 3: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 4: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 5: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 6: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 7: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 8: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 9: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 10: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 11: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 12: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 13: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 14: ['<MASK>', '<MASK>', '<MASK>'] </s>"
    bart_input = prompt + user_input + placeholder
    ids = kw_tokenizer(bart_input, return_tensors="pt").input_ids.to(device)
    generated_ids = kw_model.generate(
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
    preds = [kw_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids] 
    return str(preds).replace('.',',')

def get_rhyme(word, N):
  header = {'User-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10) AppleWebKit/600.1.25 (KHTML, like Gecko) Version/8.0 Safari/600.1.25'}
  url = "https://www.wordhippo.com/what-is/words-that-rhyme-with/" + word + ".html"
  page = requests.get(url, headers=header)
  print(page)
  soup = BeautifulSoup(page.content, 'html.parser')
  rhymes = [a.getText() for a in soup.select_one('div.relatedwords').find_all("a")]
  return rhymes[:N]

def rhyme_gen(sentence):
    words = []
    initial_rhyming_lines = [0,1,4,5,8,9,12]
    countin_rhyming_lines = [2,3,6,7,10,11,13]
    for i in range(7):
        rhy_words = []
        word = sentence[initial_rhyming_lines[i]][2]
        candidates = get_rhyme(word, N=30)
        replace_word = sentence[countin_rhyming_lines[i]][2]
        mask_input =  str(sentence).replace(replace_word,nlp.tokenizer.mask_token)
        result = nlp(mask_input, top_k = 5000)
        tokens = [res['token_str'] for res in result]
        found = False
        for t in tokens:
            if t in candidates:
                rhy_words.append(t)
                found = True
                if len(rhy_words) == 3 :
                    break
        
        if not found:
            candidates = [' '+c for c in candidates]
            result = nlp(mask_input, targets = candidates)
            tokens = [res['token_str'] for res in result]
            for t in tokens:
                if t in candidates:
                    rhy_words.append(t)
                    found = True
                    break
            if not found : #if not found in masking outputs use random token to replace word, change 2 to randint
                rhy_words.append(candidates[2])
                t = candidates[2]
        st.write('Original Word : ' + replace_word + "   " + "Candidates : " + str(rhy_words))
        sentence = str(sentence).replace(replace_word,rhy_words[0])
        sentence = eval(sentence)
    
    return sentence, words

def generate_sonnet(story_line, title, enforce_keywords=False):
    beam_size=20
    previous = ''
    enforce_keywords = enforce_keywords
    for kws in tqdm(story_line):
        success=False
        n_sample = 30
        while success != True:
            #print(kws)
            rhyme_word = kws[-1]
            prefix =  '''Keywords: ''' + '; '.join(kws) +'. Sentence in reverse order: '
            prompt = '''<|startoftext|> Title: ''' + title + ' ' + previous + prefix + rhyme_word
            p_state, n_syllables = get_phones(rhyme_word)
            result_list = []
            i=0
            prompts, all_states, all_n_sys, all_keywords = get_valid_samples(prompt,p_state, n_syllables, keywords = kws[:2], n_sample=n_sample,n_cands=5)
            while i<7:
                #print(i)
                new_prompts, new_states, new_n_sys, new_keywords = [], [], [], []
                for prompt, p_state, n_syllables, keyword in zip(prompts, all_states, all_n_sys, all_keywords):
                    t_p, t_state, t_sys, t_keywords = get_valid_samples(prompt, p_state, n_syllables, keyword,n_sample=n_sample)
                    new_prompts+=t_p
                    new_states+=t_state
                    new_n_sys+=t_sys
                    new_keywords+=t_keywords
                prompts, all_states, all_n_sys, all_keywords = new_prompts, new_states, new_n_sys, new_keywords

                prompts, all_states, all_n_sys, all_keywords = myBeamSearch(prompts,all_states, all_n_sys, all_keywords, beam_size=beam_size)
                i += 1
            correct_prompts = [reverse_order(p.split('order: ')[1]) for p in prompts]
            result_list = regularBeamSearch(correct_prompts)
            #print(result_list)
            if len(result_list)!=0:
                success=True
                found = False
                if enforce_keywords:
                    for r in result_list:
                        if kws[0] in r and kws[1] in r:
                            previous = previous + r + '\n'
                            found = True
                            break
                if found == False:
                    for r in result_list:
                        if kws[0] in r or kws[1] in r:
                            previous = previous + r + '\n'
                            found = True
                            break
                if found == False:
                    previous = previous + result_list[0]+'\n'
                    n_sample = n_sample*3

        
        print(previous + '\n')
        return previous
        

st.title("Zero-shot Sonnet Generation with Discourse-level Planning and Aesthetics Features")
write_here = "A blue lake"
title = st.text_input("Enter a title to generate the sonnet.", write_here)
keyword = generate_kws(title)
kw_clean = keyword.replace('"', ' ')
kw_clean = kw_clean.replace(':', ' ')
kw_clean = kw_clean.replace('Keywords', ' ')
kw_clean = eval(''.join([i for i in kw_clean if not i.isdigit()]))

with st.sidebar:
    st.header('Zest : A Zero-Shot Sonnet Generation Model')
    st.write('Zest does not require training on any poetic data. It consists of four components: content planning, rhyme pairing, polishing for aesthetics, and final decoding. The first three steps \
    provide salient points for the sketch of a sonnet.The last step is responsible for “translating” the sketch into well-formed sonnets.')
    st.image('arch.png')

if "Generate Keywords" not in st.session_state:
    st.session_state["Generate Keywords"] = False

if "Generate Rhyme Words" not in st.session_state:
    st.session_state["Generate Rhyme Words"] = False

if "Add Imagery" not in st.session_state:
    st.session_state["Add Imagery"] = False

if "Generate Sonnet" not in st.session_state:
    st.session_state["Generate Sonnet"] = False

if st.button("Generate Keywords"):
    #st.text_area('generated kewords',value=generate(review), key='keyword1')
    st.session_state["Generate Keywords"] = not st.session_state["Generate Keywords"]

st.subheader('Keyword Generation')

    
if st.session_state["Generate Keywords"] :  
    col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1,1])
    with col2:
        st.text_input(value = 'Keywords 1', label = "kw1", label_visibility = 'collapsed')
        st.text_input ( value = 'Keywords 2', label = "kw2", label_visibility = 'collapsed')
        st.text_input ( value = 'Keywords 3', label = "kw3", label_visibility = 'collapsed')
        st.text_input(value = 'Keywords 4', label = "kw4", label_visibility = 'collapsed')
        st.text_input ( value = 'Keywords 5', label = "kw5", label_visibility = 'collapsed')
        st.text_input ( value = 'Keywords 6', label = "kw6", label_visibility = 'collapsed')
        st.text_input(value = 'Keywords 7', label = "kw7", label_visibility = 'collapsed')
        st.text_input ( value = 'Keywords 8', label = "kw8", label_visibility = 'collapsed')
        st.text_input ( value = 'Keywords 9', label = "kw9", label_visibility = 'collapsed')
        st.text_input(value = 'Keywords 10', label = "kw10", label_visibility = 'collapsed')
        st.text_input ( value = 'Keywords 11', label = "kw11", label_visibility = 'collapsed')
        st.text_input ( value = 'Keywords 12', label = "kw12", label_visibility = 'collapsed')
        st.text_input(value = 'Keywords 13', label = "kw13", label_visibility = 'collapsed')
        st.text_input(value = 'Keywords 14', label = "kw13", label_visibility = 'collapsed')
    
    

    with col3:
        st.text_input ( value = kw_clean[0][0], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[1][0], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[2][0], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[3][0], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[4][0], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[5][0], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[6][0], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[7][0], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[8][0], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[9][0], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[10][0], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[11][0], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[12][0], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[13][0], label = "", label_visibility = 'collapsed')

    with col4:
        st.text_input ( value = kw_clean[0][1], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[1][1], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[2][1], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[3][1], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[4][1], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[5][1], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[6][1], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[7][1], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[8][1], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[9][1], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[10][1], label = "", label_visibility = 'collapsed', disabled = True)
        st.text_input ( value = kw_clean[11][1], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[12][1], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[13][1], label = "", label_visibility = 'collapsed')

    with col5:
        st.text_input ( value = kw_clean[0][2], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[1][2], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[2][2], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[3][2], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[4][2], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[5][2], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[6][2], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[7][2], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[8][2], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[9][2], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[10][2], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[11][2], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[12][2], label = "", label_visibility = 'collapsed')
        st.text_input ( value = kw_clean[13][2], label = "", label_visibility = 'collapsed')



    # edited_keyword = st.text_area('For each piece of news or stories, we train a title-tokeywords framework that predicts the outline.\
    # To this end, we first extract three most salient words per line using the RAKE (Rose et al., 2010) algorithm, which is a domain-independent keyword extraction technique',value=kw_clean)
    if st.button('Get Rhyme Words'):
        st.session_state["Generate Rhyme Words"] = not st.session_state["Generate Rhyme Words"]


st.subheader('Rhyming Module')
if st.session_state["Generate Keywords"] and st.session_state["Generate Rhyme Words"]:
    rhyme = rhyme_gen(kw_clean)
    cols2, cols3 = st.columns([1,1])
    with cols2:
        st.text_input(value = 'Keyword 1', label = "kw1", label_visibility = 'collapsed')
        st.text_input ( value = 'Keyword 2', label = "kw2", label_visibility = 'collapsed')
        st.text_input ( value = 'Keyword 3', label = "kw3", label_visibility = 'collapsed')
        st.text_input(value = 'Keyword 4', label = "kw4", label_visibility = 'collapsed')
        st.text_input ( value = 'Keyword 5', label = "kw5", label_visibility = 'collapsed')
        st.text_input ( value = 'Keyword 6', label = "kw6", label_visibility = 'collapsed')
        st.text_input(value = 'Keyword 7', label = "kw7", label_visibility = 'collapsed')
        st.text_input ( value = 'Keyword 8', label = "kw8", label_visibility = 'collapsed')
        st.text_input ( value = 'Keyword 9', label = "kw9", label_visibility = 'collapsed')
        st.text_input(value = 'Keyword 10', label = "kw10", label_visibility = 'collapsed')
        st.text_input ( value = 'Keyword 11', label = "kw11", label_visibility = 'collapsed')
        st.text_input ( value = 'Keyword 12', label = "kw12", label_visibility = 'collapsed')
        st.text_input(value = 'Keyword 13', label = "kw13", label_visibility = 'collapsed')
        st.text_input(value = 'Keyword 14', label = "kw13", label_visibility = 'collapsed')
    
    

    with cols3:
        st.text_input ( value = rhyme[0][0], label = "", label_visibility = 'collapsed')
        st.text_input ( value = rhyme[0][1], label = "", label_visibility = 'collapsed')
        st.text_input ( value = rhyme[0][2], label = "", label_visibility = 'collapsed')
        st.text_input ( value = rhyme[0][3], label = "", label_visibility = 'collapsed')
        st.text_input ( value = rhyme[0][4], label = "", label_visibility = 'collapsed')
        st.text_input ( value = rhyme[0][5], label = "", label_visibility = 'collapsed')
        st.text_input ( value = rhyme[0][6], label = "", label_visibility = 'collapsed')
        st.text_input ( value = rhyme[0][7], label = "", label_visibility = 'collapsed')
        st.text_input ( value = rhyme[0][8], label = "", label_visibility = 'collapsed')
        st.text_input ( value = rhyme[0][9], label = "", label_visibility = 'collapsed')
        st.text_input ( value = rhyme[0][10], label = "", label_visibility = 'collapsed')
        st.text_input ( value = rhyme[0][11], label = "", label_visibility = 'collapsed')
        st.text_input ( value = rhyme[0][12], label = "", label_visibility = 'collapsed')
        st.text_input ( value = rhyme[0][13], label = "", label_visibility = 'collapsed')

    if st.button('Add Imagery'):
        st.session_state["Add Imagery"] = not st.session_state["Add Imagery"]

st.subheader('Polishing Module - Imagery')
if st.session_state["Generate Keywords"] and st.session_state["Generate Rhyme Words"] and st.session_state["Add Imagery"]:
    location_dict ={}
    for i, keywords in enumerate(rhyme[0]):
        w1, w2, _ = keywords
        ent = pos(w1)[0]
        if ent.pos_ == 'NOUN':
            location_dict[str(ent)] = [i,0]
            continue
        ent = pos(w2)[0]
        if ent.pos_ == 'NOUN':
            location_dict[str(ent)] = [i,1]

    st.write('Select NOUNs to replace with imagery:')
    sample = random.sample(location_dict.keys(), 5)
    option_1 = st.checkbox(sample[0])
    option_2 = st.checkbox(sample[1])
    option_3 = st.checkbox(sample[2])
    option_4 = st.checkbox(sample[3])

    if st.button('Generate Sonnet'):
        st.session_state["Generate Sonnet"] = not st.session_state["Generate Sonnet"]
    
    

# if st.session_state["Generate Keywords"] and st.session_state["Generate Rhyme Words"] and st.session_state["Generate Sonnet"]:
#     sonnet = st.text_area('The generated sonnet', value = generate_sonnet(rhyme, title))


        
        

