#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPTNeoModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pronouncing
from transformers import Trainer, TrainingArguments

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


import logging, sys
logging.disable(sys.maxsize)


# In[4]:


# Set the random seed to a fixed value to get reproducible results 
torch.manual_seed(42)
# Download the pre-trained GPT-Neo model's tokenizer
# Add the custom tokens denoting the beginning and the end 
# of the sequence and a special token for padding
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", 
                                          bos_token="<|startoftext|>",
                            eos_token="<|endoftext|>",
                            pad_token="<|pad|>")


# In[5]:


# Download the pre-trained GPT-Neo model and transfer it to the GPU
model = GPTNeoForCausalLM.from_pretrained("news-gpt-neo-1.3B-keywords-line-by-line-reverse/checkpoint-15000").cuda()
# Resize the token embeddings because we've just added 3 new tokens 
model.resize_token_embeddings(len(tokenizer))


# ### rhymes

# In[6]:


import pronouncing


# In[7]:


def get_stress(phone):
    stress = []
    for s in phone.split():
        if s[-1].isdigit():
            if s[-1] == '2':
                stress.append(0)
            else:
                stress.append(int(s[-1]))
    return stress


# In[8]:


def alternating(stress):
    #Check if the stress and unstress are alternating
    check1 = len(set(stress[::2])) <= 1 and (len(set(stress[1::2])) <= 1)
    check2 = len(set(stress)) == 2 if len(stress) >=2 else True
    return (check1 and check2)
                                            


# In[9]:


def get_phones(rhyme_word):
    phone = pronouncing.phones_for_word(rhyme_word)[0]
    stress = get_stress(phone)
    p_state = stress[0]
    n_syllables = len(stress)
    return p_state, n_syllables


# ### Generation

# In[10]:


from torch import Tensor
from torch.nn import functional as F


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


# In[11]:


#generate a complete word instead of subwords
def generate_next_word(input_ids1, temperature = 0.85, topk = 100, device = 'cuda:0'):
        current_word = 0
        for _ in range(10):
            outputs1 = model(input_ids1)
            next_token_logits1 = outputs1[0][:, -1, :]
            next_token_logits1 = top_k_top_p_filtering(next_token_logits1, top_k=topk)
            logit_zeros = torch.zeros(len(next_token_logits1)).cuda()
            #logit_zeros = torch.zeros(len(next_token_logits1), device=device)

            next_token_logits = next_token_logits1 * temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            #unfinished_sents = torch.ones(1, dtype=torch.long, device=device)
            unfinished_sents = torch.ones(1, dtype=torch.long).cuda()
            tokens_to_add = next_tokens * unfinished_sents + tokenizer.pad_token_id * (1 - unfinished_sents)

            if tokenizer.eos_token_id in next_tokens[0]:
                input_ids1 = torch.cat([input_ids1, tokens_to_add.unsqueeze(-1)], dim=-1)
                return '', True

            if tokenizer.decode(tokens_to_add[0])[0] == ' ':
                if current_word ==1:
                    return tokenizer.decode(input_ids1[0]).split()[-1], False
                current_word += 1
            input_ids1 = torch.cat([input_ids1, tokens_to_add.unsqueeze(-1)], dim=-1)
        return None


# In[12]:


def get_valid_samples(prompt, p_state, n_syllables, keywords):
    states = []
    all_n_syl = []
    
    prompts = []
    all_keywords = [] 
    #insert the keyword whenever possible
    for source_word in keywords:
        phone = pronouncing.phones_for_word(source_word)[0]
        stress = get_stress(phone)
        if not alternating(stress):
            continue

        #if the word is single syllable and can be either stressed or unstressed, flag = True
        flag = check_either_stress(stress, source_word)

        if stress[-1] == 1- p_state or flag:
            states.append(stress[0])
            all_n_syl.append(n_syllables+len(stress))
            prompts.append(prompt+ ' ' + source_word )
            copy = keywords.copy()
            copy.remove(source_word)
            all_keywords.append(copy)    
    
    #The normal process of decoding
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    tokens = []
    while len(tokens) < 3:
        token, eos = generate_next_word(input_ids)
        if (token not in tokens) and (token not in keywords):
            #print(token, tokens)
            try:
                phone = pronouncing.phones_for_word(token)[0]
                stress = get_stress(phone)
                if not alternating(stress):
                    continue

                #if the word is single syllable and can be either stressed or unstressed, flag = True
                flag = check_either_stress(stress, token)

                if stress[-1] == 1- p_state or flag:
                    tokens.append(token)
                    states.append(stress[0])
                    all_n_syl.append(n_syllables+len(stress))
                    prompts.append(prompt+ ' ' + token )
                    all_keywords.append(keywords)
            except:
                continue

    return prompts, states, all_n_syl, all_keywords


# In[13]:


def check_either_stress(stress, source_word, loose = False):
    if loose:
        return len(stress) == 1
    if len(stress) == 1 and len(pronouncing.phones_for_word(source_word))>1:
                    phone0 = pronouncing.phones_for_word(source_word)[0]
                    phone1 = pronouncing.phones_for_word(source_word)[1]
                    stress0 = [int(s[-1]) for s in phone0.split() if s[-1].isdigit()]
                    stress1 = [int(s[-1]) for s in phone1.split() if s[-1].isdigit()]
                    if stress0+stress1 ==1 and stress0*stress1 == 0:
                        return True
    return False


# In[56]:





# In[18]:


def reverse_order(line):
    line = line.replace(', ', ' , ')
    words = line.split()
    return ' '.join(reversed(words)).replace(' , ', ', ')


# In[43]:


device = 'cuda:0'
score_model = model
def myBeamSearch(true_beams, beam_size = 5):
    BeamScorer = {}
    for sentence in true_beams:
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])

        tensor_input = tensor_input.to(device)
        loss = score_model(tensor_input, labels=tensor_input)
        avg_lp = torch.tensor(-loss[0].item()/len(tokenize_input))
        BeamScorer[sentence] = avg_lp
    BeamScorer = {k: v for k, v in sorted(BeamScorer.items(), key=lambda x: x[1], reverse=True)}
    return list(BeamScorer.keys())[:beam_size]


# In[50]:


beam_size = 5
def gen_recursion(prompt, p_state, n_syllables, keywords):
    global result_list
    '''I modified this criterion to speed up the example.
    I suggest to add non-repeat-unigram (= 3) and keyword checking
    '''
    if n_syllables >= 5: 
        line = prompt.split(': ')[-1]
        reversed_words = reverse_order(line)
        reversed_words = reversed_words
        result_list.append(reversed_words)
        if len(result_list)>=beam_size:
            result_list = myBeamSearch(result_list, beam_size = beam_size)
            #print(result_list)
        return result_list
    prompts, states, all_n_sys, all_keywords = get_valid_samples(prompt,p_state, n_syllables, keywords)
    for prompt,p_state, n_syllables, keyword in zip(prompts, states, all_n_sys, all_keywords):
        gen_recursion(prompt,p_state, n_syllables, keywords)


# In[56]:


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


# In[54]:


example_title = 'The Four Seasons'

previous = ''
for kws in four_seasons_story_line[:4]:
    print(kws)
    rhyme_word = kws[-1]
    prefix =  '''Keywords: ''' + '; '.join(kws) +'. Sentence in reverse order: '
    prompt = '''<|startoftext|> Title: ''' + example_title + ' ' previous + prefix + rhyme_word
    p_state, n_syllables = get_phones(rhyme_word)
    result_list = []
    #to add hard constraints, specify keywords, otherwise use keywords = []
    gen_recursion(prompt, p_state, n_syllables, keywords = ['air','heat'])
    print(result_list)
    
    previous = previous + result_list[0] + ', '


# In[55]:


previous

