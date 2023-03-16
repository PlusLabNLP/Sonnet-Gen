
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPTNeoModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import pipeline
import pronouncing
import requests
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from fairseq.models.bart import BARTModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive
cfg.device = "cpu"

import random
import warnings
warnings.filterwarnings('ignore')

#from decoding import *

import spacy
pos = spacy.load('en_core_web_sm')

from torch import cuda
device = 0 if cuda.is_available() else 'cpu'

@st.cache_resource
def load_keyword_model():
    tokenizer = AutoTokenizer.from_pretrained("FigoMe/sonnet_keyword_gen")
    model = AutoModelForSeq2SeqLM.from_pretrained("FigoMe/sonnet_keyword_gen").to(device)
    return model, tokenizer

@st.cache_resource
def load_decoding_model():
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", 
                                          bos_token="<|startoftext|>",
                            eos_token="<|endoftext|>",
                            pad_token="<|pad|>")
    
    # Download the pre-trained GPT-Neo model and transfer it to the GPU
    model = GPTNeoForCausalLM.from_pretrained("FigoMe/news-gpt-neo-1.3B-keywords-line-by-line-reverse").to(device)
    # Resize the token embeddings because we've just added 3 new tokens 
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

@st.cache_resource
def load_gpt_model():
    tokenizer  = AutoTokenizer.from_pretrained('gpt2-large')
    gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2-large')
    gpt2_model = gpt2_model.to(device)
    gpt2_model.eval()
    return gpt2_model, tokenizer

@st.cache_resource 
def load_simile_model():
    tok = AutoTokenizer.from_pretrained("facebook/bart-large")
    datadir = 'simile'
    cpdir = datadir+'/'
    bart = BARTModel.from_pretrained(cpdir,checkpoint_file='checkpoint-simile/checkpoint_best.pt',data_name_or_path=datadir).to(device)
    bart.eval()
    return bart, tok

@st.cache_resource
def load_imagery_model():
    model_file =  '/local1/asuvarna31/Sonnet-Gen/demo/pretrained_models/reverse_comet_1e-05_adam_32_20000.pickle'
    opt, state_dict = interactive.load_model_file(model_file)
    data_loader, text_encoder = interactive.load_data("conceptnet", opt)

    n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
    n_vocab = len(text_encoder.encoder) + n_ctx

    model = interactive.make_model(opt,  40543, 29, state_dict)
    return data_loader, text_encoder, model, opt

keyword_model, keyword_tokenizer = load_keyword_model()
sonnet_model, sonnet_tokenizer = load_decoding_model()
gpt_model, gpt_tokenizer = load_gpt_model()
smile_model, simile_tokenizer = load_simile_model()
data_loader, text_encoder, imagery_model, opt = load_imagery_model()


nlp = pipeline("fill-mask", model=keyword_model, tokenizer = keyword_tokenizer, device=device)


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

def check_meter(adj, vehicle):
    phone = pronouncing.phones_for_word(adj)[0]
    stress_adj = get_stress(phone)
    vehicle = vehicle.strip(',.<>')
    try:
        if len(vehicle.split())==1:
            phone = pronouncing.phones_for_word(vehicle)[0]
            stress_vehicle = get_stress(phone)
        else:
            stress_vehicle = []
            for word in vehicle.split():
                phone = pronouncing.phones_for_word(word)[0]
                stress_vehicle += get_stress(phone)
        #assume 'like' can be either stressed or unstressed
        if stress_vehicle[0]==stress_adj[-1] and alternating(stress_vehicle):
            return True
    except:
        pass
    return False

def is_noun_phrase(vehicle):
    if vehicle.startswith('a '):
        vehicle = vehicle.replace('a ','')
    doc = pos(vehicle)
    flag = False
    for ent in doc:
        if ent.pos_== 'NOUN' or ent.pos_=='PROPN':
            return True
    return False

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


def generate_next_word(input_ids1, temperature = 0.85, topk = 100, n_sample=10, device = 'cuda:0'):
    current_word = 0
    original = sonnet_tokenizer.decode(input_ids1[0])
    for _ in range(1):
        outputs1 = sonnet_model(input_ids1)
        #print(outputs1)
        next_token_logits1 = outputs1[0][:, -1, :]
        next_token_logits1 = top_k_top_p_filtering(next_token_logits1, top_k=topk)
        logit_zeros = torch.zeros(len(next_token_logits1)).cuda()
        #logit_zeros = torch.zeros(len(next_token_logits1), device=device)

        next_token_logits = next_token_logits1 * (1/ temperature)
        probs = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=n_sample).squeeze(1)
        #unfinished_sents = torch.ones(1, dtype=torch.long, device=device)
        unfinished_sents = torch.ones(1, dtype=torch.long).cuda()
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
        '''
        if tokenizer.decode(tokens_to_add[0])[0] == ' ':
            if current_word ==1:
                return tokenizer.decode(input_ids1[0]).split()[-1], False
            current_word += 1
        input_ids1 = torch.cat([input_ids1, tokens_to_add.unsqueeze(-1)], dim=-1)
        '''



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


softmax = torch.nn.Softmax(dim=1)
def sample_prompts(prompts,previous='', temperature = 1):
    BeamScorer = {}
    for sentence in prompts:
        loss = score_gpt2(previous+sentence)
        BeamScorer[sentence] = [loss]
    p = BeamScorer.values()
    p = torch.tensor(list(p))*(1/temperature)
    try:
        p = p.squeeze(1)
    except:
        pass
    p_softmax = torch.nn.functional.softmax(p)
    print(p_softmax)
    index = torch.multinomial(p_softmax,num_samples=len(prompts))
    print(f'selected index: {index}')
    new_prompts = [prompts[i] for i in index]
    return new_prompts

def score_gpt2(sentence, normalize = True):
	'''
	The default setting is to normalize because we won't face the issue mentioned in function "score".
	'''
	tokens_tensor = gpt_tokenizer.encode(sentence, add_special_tokens=False, return_tensors="pt")[0].cuda()
	with torch.no_grad():
		loss = gpt_model(tokens_tensor, labels=tokens_tensor)[0]
	if normalize:
		return loss/len(tokens_tensor)
	else:
		return loss

def myBeamSearch(prompts, all_states, all_n_sys, all_keywords, beam_size = 5,enforce_keywords=True):
    BeamScorer = {}
    return_seq, return_stt, return_sys, return_key = [], [], [], []
    
    if (not enforce_keywords) or len(all_keywords)==0:
        for sentence, p_state, n_sys, keywords in zip(prompts, all_states, all_n_sys, all_keywords):
            loss = score(sentence)
            BeamScorer[sentence] = [loss, p_state, n_sys, keywords]
        answers = sorted(BeamScorer.items(), key=lambda x: x[1], reverse=False)
    else:
        min_remaining = min([len(x) for x in all_keywords])
        for sentence, p_state, n_sys, keywords in zip(prompts, all_states, all_n_sys, all_keywords):
            #start with fewer keywords remaining
            if len(keywords) == min_remaining:
                loss = score(sentence)
                BeamScorer[sentence] = [loss, p_state, n_sys, keywords]
        answers = sorted(BeamScorer.items(), key=lambda x: x[1], reverse=False)
        BeamScorer={}
        for sentence, p_state, n_sys, keywords in zip(prompts, all_states, all_n_sys, all_keywords):
            #then
            if len(keywords) == min_remaining+1:
                loss = score(sentence)
                BeamScorer[sentence] = [loss, p_state, n_sys, keywords]
        answers += sorted(BeamScorer.items(), key=lambda x: x[1], reverse=False)
        BeamScorer={}
        for sentence, p_state, n_sys, keywords in zip(prompts, all_states, all_n_sys, all_keywords):
            #last, most keywords remaining
            if len(keywords) == min_remaining+2:
                loss = score(sentence)
                BeamScorer[sentence] = [loss, p_state, n_sys, keywords]
        answers += sorted(BeamScorer.items(), key=lambda x: x[1], reverse=False)
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

def score(sentence, normalize = True):
	'''
	Score a single sentence using the plan-to-lyrics model.
	The recommended setting is to NOT normalize, because the input sentence is very long: it contains the title, planed keywords, and previously generated lines. 
	In addition, the candidate sentences contain the same prefix (i.e., the title, planed keywords, and previously generated lines) and only differ in the currently generated line.
	Normaling means dividing the loss by a large factor which may result in similarity accross different candidate sentences.
	'''
	tokens_tensor = sonnet_tokenizer.encode(sentence, add_special_tokens=False, return_tensors="pt")[0].cuda()
	with torch.no_grad():
		loss = sonnet_model(tokens_tensor, labels=tokens_tensor)[0]
	if normalize:
		return loss/len(tokens_tensor)
	else:
		return loss

def get_stress_phrase(phrase):
    words = phrase.split()
    stress=[]
    for source_word in words:
        phone = pronouncing.phones_for_word(source_word)[0]
        stress+= get_stress(phone)
    return stress

single_character_word = ['i','a']
forbidden_words = ['dona','er','ira','ia',"'s","'m","hmm","mm"]

def get_valid_samples(prompt, p_state, n_syllables, keywords, n_sample=30, n_cands=5):
    #if n_syllables == 10 or n_syllables==11:
    if n_syllables == 10 and len(keywords)==0:
        return [prompt], [p_state], [n_syllables], [keywords]
    elif n_syllables > 10:
        return [], [], [],[]
    states = []
    all_n_syl = []
    
    prompts = []
    all_keywords= [] 
    #insert the keyword whenever possible
    for source_word in keywords:
        stress = get_stress_phrase(source_word)
        #if not alternating(stress):
            #continue

        #if the word is single syllable and can be either stressed or unstressed, flag = True
        flag = check_either_stress(stress, source_word)

        if (stress[-1] == 1- p_state or flag) and (n_syllables+len(stress)<=10):
            states.append(stress[0])
            all_n_syl.append(n_syllables+len(stress))
            #print(source_word)
            prompts.append(prompt+ ' ' +reverse_order(source_word))
            copy = keywords.copy()
            copy.remove(source_word)
            all_keywords.append(copy)    
    
    #The normal process of decoding
    input_ids = sonnet_tokenizer(prompt, return_tensors='pt').input_ids.cuda()
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
            if n_syllables+len(stress)<=10:
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

def getloss(input_e1, input_e2, relation, model,prnt = False):
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

def getPred(input_event, relation, prnt = True, sampling_algorithm = 'beam-2'):
    sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)
    outputs = interactive.get_conceptnet_sequence(input_event, imagery_model, sampler, data_loader, text_encoder, relation, prnt)
    return outputs

def get_imagery(samples, story_line):
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
    #samples = random.sample(location_dict.keys(),N)
    relations = ['SymbolOf']
    score_dict = {}
    replace_dict = {}
    polished_lines = []
    flatten_list = [j for sub in story_line for j in sub]
    for ent in samples:
        result = getPred(ent, relation=relations, sampling_algorithm = 'topk-10', prnt = False)[relations[0]]['beams']
        for i in range(len(result)):
            if result[i] not in flatten_list:
                result = result[i]
                break
        score_dict[ent] = getloss(ent, result, 'SymbolOf', imagery_model,prnt = False)
        replace_dict[ent] = result
        return replace_dict



def get_phones(rhyme_word):
    phone = pronouncing.phones_for_word(rhyme_word)[0]
    stress = get_stress(phone)
    p_state = stress[0]
    n_syllables = len(stress)
    return p_state, n_syllables


def generate_kws(user_input, _kw_tokenizer, _kw_model):
    prompt = "Generate keywords for the title: "
    placeholder = ". Keywords 1: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 2: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 3: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 4: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 5: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 6: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 7: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 8: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 9: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 10: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 11: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 12: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 13: ['<MASK>', '<MASK>', '<MASK>'] . Keywords 14: ['<MASK>', '<MASK>', '<MASK>'] </s>"
    bart_input = prompt + user_input + placeholder
    ids = _kw_tokenizer(bart_input, return_tensors="pt").input_ids.to(device)
    generated_ids = _kw_model.generate(
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
    preds = [_kw_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids] 
    return str(preds).replace('.',',')

@st.cache_data
def get_rhyme(word, N):
  header = {'User-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10) AppleWebKit/600.1.25 (KHTML, like Gecko) Version/8.0 Safari/600.1.25'}
  url = "https://www.wordhippo.com/what-is/words-that-rhyme-with/" + word + ".html"
  page = requests.get(url, headers=header)
  #print(page)
  soup = BeautifulSoup(page.content, 'html.parser')
  rhymes = [a.getText() for a in soup.select_one('div.relatedwords').find_all("a")]
  return rhymes[:N]

def isPronouncing(word):
    phone = pronouncing.phones_for_word(word)
    if len(phone) == 0:
        return False
    else:
        return True

def rhyme_gen(sentence):
    words = {}
    initial_rhyming_lines = [0,1,4,5,8,9,12]
    countin_rhyming_lines = [2,3,6,7,10,11,13]
    for i in range(7):
        rhy_words = []
        scores = {}
        word = sentence[initial_rhyming_lines[i]][2]
        candidates = get_rhyme(word, N=5)
        candidates = list(filter(lambda x : isPronouncing(x), candidates))
        replace_word = sentence[countin_rhyming_lines[i]][2]
        # mask_input =  str(sentence).replace(replace_word,nlp.tokenizer.mask_token)
        # result = nlp(mask_input, targets= candidates)
        # for res in result:
        #     scores[res['token_str']] = res['score']
        #st.write('Original Word : ' + word + "   " + "Candidates : " + str(candidates[:5]))
        words[word] = candidates
        sentence = str(sentence).replace(replace_word, random.choice(candidates))
        sentence = eval(sentence)
    
    return sentence, words

@st.cache_data
def simile_vehicle(inp, t=0.8):
        simile_phrases=[]
        prefix = inp+' like'
        inputs = [prefix]*10
        l=len(simile_tokenizer(prefix)[0])
        hypotheses_batch = smile_model.sample(inputs,sampling=True, sampling_topk=5, temperature=t, max_len_b=l+2, min_len=l)
        for hypothesis in hypotheses_batch:
            vehicle = hypothesis.split(' like ')[1].split('<')[0].lower()
            vehicle = vehicle.replace('like ','')
            print(vehicle)
            if is_noun_phrase(vehicle) and check_meter(inp, vehicle):
                simile_phrases.append(' '.join([inp, 'like', vehicle]))
                print('yes')
            if vehicle.startswith('a ') or vehicle.startswith('an ') or vehicle.startswith('the '):
                vehicle = vehicle.replace('a ','')
                vehicle = vehicle.replace('an ','')
                vehicle = vehicle.replace('the ','')
                if is_noun_phrase(vehicle) and check_meter(inp, vehicle):
                    simile_phrases.append(' '.join([inp, 'like', vehicle]))
                    print('yes')
        return list(set(simile_phrases))

def generate_sonnet(story_line, title, enforce_keywords=True):
    print(story_line)
    beam_size=20
    previous = ''
    enforce_keywords = enforce_keywords
    for kws in tqdm(story_line):
        success=False
        n_sample = 30
        while success != True:
            print(kws)
            rhyme_word = kws[-1]
            prefix =  '''Keywords: ''' + '; '.join(kws) +'. Sentence in reverse order: '
            prompt = '''<|startoftext|> Title: ''' + title + ' ' + ','.join(previous.split(',')[-3:]) + prefix + rhyme_word
            #prompt = '''<|startoftext|> Title: ''' + example_title + ' ' + prefix + rhyme_word
            try:
                p_state, n_syllables = get_phones(rhyme_word)
            except:
                st.write(f'Failed to find the pronounciation of {rhyme_word}, assuming it has one stressed syllable')
                p_state, n_syllables = get_phones('one')
                
            result_list = []
            i=0
            #print(1)
            prompts, all_states, all_n_sys, all_keywords = get_valid_samples(prompt,p_state, n_syllables, keywords = kws[:2], n_sample=n_sample,n_cands=5)
            while i<7:
                #print(i)
                new_prompts, new_states, new_n_sys, new_keywords = [], [], [], []
                for prompt, p_state, n_syllables, keyword in zip(prompts, all_states, all_n_sys, all_keywords):
                    #print(2)
                    t_p, t_state, t_sys, t_keywords = get_valid_samples(prompt, p_state, n_syllables, keyword,n_sample=n_sample)
                    new_prompts+=t_p
                    new_states+=t_state
                    new_n_sys+=t_sys
                    new_keywords+=t_keywords
                prompts, all_states, all_n_sys, all_keywords = new_prompts, new_states, new_n_sys, new_keywords
                prompts, all_states, all_n_sys, all_keywords = myBeamSearch(prompts,all_states, all_n_sys, all_keywords, beam_size=beam_size, enforce_keywords=enforce_keywords)
                i += 1
            if len(prompts)==0:
                if n_sample>300:
                    print('Failed to generate valid samples. Please try re-generation for this line.')
                    previous += '   ,' + '\n'
                    break
                n_sample = n_sample*3

            else:
                correct_prompts = [reverse_order(p.split('order: ')[1]) for p in prompts]
                #print(correct_prompts)
                result_list = sample_prompts(correct_prompts, previous)
                
                success=True
                found = False 
                for r in result_list:
                    if kws[0] in r or kws[1] in r:
                        previous = previous + r + ',' + '\n'
                        found = True
                        st.write(r)
                        break
                if found == False:
                        previous = previous + result_list[0]+',\n'
                        n_sample = n_sample*3
                        st.write(result_list[0]+ ',\n')
    return previous

st.title("Zero-shot Sonnet Generation with Discourse-level Planning and Aesthetics Features")
write_here = "A classy lake"
title = st.text_input("Enter a title to generate the sonnet.", write_here)
with st.sidebar:
    st.header('Zest : A Zero-Shot Sonnet Generation Model')
    st.write("This is a web demo for Tian & Peng 2022. All the user inputs will be used for academic purposes. This demo is powered by finetuned GPT2 model and please do not go back to previous modules once you have clicked the Generate Sonnet button unless the sonnet is completely generated. ")
    need_polish = st.radio("Do you want to add Imagery and Similes in your sonnet ?", ('Yes', 'No'))

if "Generate Keywords" not in st.session_state:
    st.session_state["Generate Keywords"] = False

if "Generate Rhyme Words" not in st.session_state:
    st.session_state["Generate Rhyme Words"] = False

if "Polish" not in st.session_state:
    st.session_state["Polish"] = False

if "Generate Sonnet" not in st.session_state:
    st.session_state["Generate Sonnet"] = False

if st.button("Generate Keywords"):
    #st.text_area('generated kewords',value=generate(review), key='keyword1')
    st.session_state["Generate Keywords"] = not st.session_state["Generate Keywords"]

st.subheader('Keyword Generation')
keyword = generate_kws(title, keyword_tokenizer, keyword_model)
kw_clean = keyword.replace('"', ' ')
kw_clean = kw_clean.replace(':', ' ')
kw_clean = kw_clean.replace("n’t", "not")
kw_clean = kw_clean.replace('Keywords', ' ')
kw_clean = eval(''.join([i for i in kw_clean if not i.isdigit()]))
dictionary = {}
line = []
    
if st.session_state["Generate Keywords"] :  
    line1, line2, line3 = [], [], []
    
    #with st.form(key='columns_in_form'):
    for i in range(0, 14):
        cols = st.columns(4)
        cols[0].text('Keywords'+f'{i}')
        dictionary["line{0}_word1".format(i)] = cols[1].text_input(value=kw_clean[i][0], label = "coffee{0}".format(i), label_visibility = 'collapsed',  disabled = False)
        dictionary["line{0}_word2".format(i)] = cols[2].text_input(value=kw_clean[i][1], label = "toffee{0}".format(i), label_visibility = 'collapsed',  disabled = False)
        dictionary["line{0}_word3".format(i)] = cols[3].text_input(value=kw_clean[i][2],label = "roffee{0}".format(i), label_visibility = 'collapsed',  disabled = False)
    
        #rhyme_get = st.form_submit_button('Get Rhyme Words')


    for i in range(0,14):
        line.append([dictionary["line{0}_word1".format(i)],dictionary["line{0}_word2".format(i)], dictionary["line{0}_word3".format(i)] ])
    
    if st.button("Generate Rhyme Words"):
        st.session_state["Generate Rhyme Words"] = not st.session_state["Generate Rhyme Words"]



st.subheader('Rhyming Module')
if st.session_state["Generate Keywords"] and st.session_state["Generate Rhyme Words"]:
    
    
    rhyme, output = rhyme_gen(line)

    with st.form(key='rhyme_cols'):
        for col in output:
            cols = st.columns(3)
            cols[0].text(col)
            cols[1].selectbox('Please select the rhyme word you want', output[col])

        polish = st.form_submit_button('Polish')


    if polish:
            st.session_state["Polish"] = not st.session_state["Polish"]
    
    

st.subheader('Polishing Modules')

if need_polish == 'No':
    new_plan = rhyme
    st.write('Please proceed to Sonnet generation as you have selected to not add imagery and simile in your sonnet')
else:
    if st.session_state["Generate Keywords"] and st.session_state["Generate Rhyme Words"] and st.session_state["Polish"]:
        adj_dict ={}
        noun_dict = {}
        for i, keywords in enumerate(rhyme):
            #if i not in polished_lines:
            w1, w2, _ = keywords
            ent = pos(w1)[0]
            if ent.pos_ == 'ADJ':
                adj_dict[str(ent)] = [i,0]
                continue
            ent = pos(w2)[0]
            if ent.pos_ == 'ADJ':
                adj_dict[str(ent)] = [i,1]

            if ent.pos_ == 'NOUN':
                noun_dict[str(ent)] = [i,0]
                continue
            if ent.pos_ == 'NOUN':
                noun_dict[str(ent)] = [i,1]
        
        new_plan = rhyme

        adjs = list(adj_dict.keys())
        if len(adjs) > 0:
            options = st.multiselect('Choose the adjective you want to polish from  dropdown', adjs)
            for word in options:
                simile = simile_vehicle(word)
                st.write('Selected Adjective', word)
                st.write('Generated Similes', simile)
                num = st.number_input('Please enter index of simile you want to choose', value=0, step=1, key=word, min_value = 0, max_value = len(simile)-1)
                new_plan = str(new_plan).replace(word,simile[num])

        nouns = list(noun_dict.keys())
        if len(nouns) > 0:
            imagery_options = st.multiselect('Choose the noun you want to polish from dropdown', nouns)
            relations = ['SymbolOf']
            score_dict = {}
            replace_dict = {}
            polished_lines = []

            for word in imagery_options:
                
                result = getPred(word, relation=relations, sampling_algorithm = 'topk-10', prnt = False)[relations[0]]['beams']
                flatten_list = [j for sub in new_plan for j in sub]
                result = list(filter(lambda x: x not in flatten_list, result))
                st.write(result[:5])
                
                st.write('Selected Noun', word)
                num_i = st.number_input('Please enter index of imagery you want to choose', value=0, step=1, key=word, min_value = 0, max_value = 4)
                new_plan = str(new_plan).replace(word,result[num_i])
                

if st.button('Generate Sonnet', help = 'Sonnet is generated using finetuned GPT2 and the decoding step may take ~15 mins'):
    st.session_state["Generate Sonnet"] = not st.session_state["Generate Sonnet"]

st.subheader('Generating the Final Sonnet')    
if st.session_state["Generate Keywords"] and st.session_state["Generate Rhyme Words"] and st.session_state["Generate Sonnet"]:
    if (type(new_plan)) != 'list':
        new_plan = eval(new_plan)
    sonnet = st.text_area('The generated sonnet', value = generate_sonnet(new_plan, title), height = 500)




        
        

