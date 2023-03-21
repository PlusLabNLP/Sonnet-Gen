from bs4 import BeautifulSoup
from transformers import pipeline
import requests
import pronouncing
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

device = 7

tokenizer = AutoTokenizer.from_pretrained("FigoMe/sonnet_keyword_gen")
model = AutoModelForSeq2SeqLM.from_pretrained("FigoMe/sonnet_keyword_gen").to(device)
nlp = pipeline("fill-mask", model=model, tokenizer = tokenizer, device=device)

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



def score_phrase(phrase_with_mask, target_word):
    if target_word[0] == " ":
        target_word_g = "Ġ" + target_word[1:]
    else:
        target_word_g = target_word
    if target_word_g in tokenizer.vocab:
        scored_phrase = nlp(phrase_with_mask, targets=target_word_g)
        return scored_phrase
    else:
        scores = []
        mask_insert_position = phrase_with_mask.find(tokenizer.mask_token)
        tokenized_target_word_g = tokenizer.tokenize(target_word)    
        while len(tokenized_target_word_g) > 0:
            token_to_insert = tokenized_target_word_g[0]
            tokenized_target_word_g.pop(0)
            scored_phrase = nlp(phrase_with_mask, targets=token_to_insert)
            scores.append(scored_phrase[0]['score'])
            if token_to_insert[0] == 'Ġ':
                mask_insert_position += len(token_to_insert) - 1
            else:
                mask_insert_position += len(token_to_insert)
            phrase_with_mask = str(scored_phrase[0]['sequence'][:mask_insert_position]) + str(tokenizer.mask_token) + str(scored_phrase[0]['sequence'][mask_insert_position:])
#         print(scores)
        sum_of_scores = 0
        for each in scores:
            sum_of_scores += each
        avg_score = sum_of_scores/len(scores)

        
        return {'score': sum_of_scores, 'token': scored_phrase[0]['token'], 'token_str': target_word}

def rhyme_gen(sentence):
    scores = {}
    result = []
    word = sentence[2]
    candidates = get_rhyme(word, N=100)
    candidates = list(filter(lambda x : isPronouncing(x), candidates))
    replace_word = word
    print(replace_word)
    mask_input =  str(sentence).replace(replace_word,nlp.tokenizer.mask_token)
    for c in candidates:
        if c != replace_word:
            result.append(score_phrase(mask_input, c))
        
    scores = {}
    for res in result:
        scores[res['token_str']] = res['score']

    sorted_scores = sorted(scores.items(), key=lambda x: -1*x[1])
    print(sorted_scores)

    
    
        

rhyme_gen(['coffee', 'shallow', 'stopped'])