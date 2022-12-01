import json
import copy

import torch

import numpy as np
import contextlib

from distutils.dir_util import mkpath

from tqdm import tqdm


def make_new_tensor_from_list(items, device_num, dtype=torch.float32):
    if device_num is not None:
        device = torch.device("cuda:{}".format(device_num))
    else:
        device = torch.device("cpu")
    return torch.tensor(items, dtype=dtype, device=device)


# is_dir look ast at whether the name we make
# should be a directory or a filename
def make_name(opt, prefix="", eval_=False, is_dir=True, set_epoch=None,
              do_epoch=True):
    string = prefix
    string += "{}-{}".format(opt.dataset, opt.exp)
    string += "/"
    string += "{}-{}-{}".format(opt.trainer, opt.cycle, opt.iters)
    string += "/"
    string += opt.model
    if opt.mle:
        string += "-{}".format(opt.mle)
    string += "/"
    string += make_name_string(opt.data) + "/"

    string += make_name_string(opt.net) + "/"
    string += make_name_string(opt.train.static) + "/"

    if eval_:
        string += make_name_string(opt.eval) + "/"
    # mkpath caches whether a directory has been created
    # In IPython, this can be a problem if the kernel is
    # not reset after a dir is deleted. Trying to recreate
    # that dir will be a problem because mkpath will think
    # the directory already exists
    if not is_dir:
        mkpath(string)
    string += make_name_string(
        opt.train.dynamic, True, do_epoch, set_epoch)
    if is_dir:
        mkpath(string)

    return string


def make_name_string(dict_, final=False, do_epoch=False, set_epoch=None):
    if final:
        if not do_epoch:
            string = "{}_{}_{}".format(
                dict_.lr, dict_.optim, dict_.bs)
        elif set_epoch is not None:
            string = "{}_{}_{}_{}".format(
                dict_.lr, dict_.optim, dict_.bs, set_epoch)
        else:
            string = "{}_{}_{}_{}".format(
                dict_.lr, dict_.optim, dict_.bs, dict_.epoch)

        return string

    string = ""

    for k, v in dict_.items():
        if type(v) == DD:
            continue
        if isinstance(v, list):
            val = "#".join(is_bool(str(vv)) for vv in v)
        else:
            val = is_bool(v)
        if string:
            string += "-"
        string += "{}_{}".format(k, val)

    return string


def is_bool(v):
    if str(v) == "False":
        return "F"
    elif str(v) == "True":
        return "T"
    return v


def generate_config_files(type_, key, name="base", eval_mode=False):
    with open("config/default.json".format(type_), "r") as f:
        base_config = json.load(f)
    with open("config/{}/default.json".format(type_), "r") as f:
        base_config_2 = json.load(f)
    if eval_mode:
        with open("config/{}/eval_changes.json".format(type_), "r") as f:
            changes_by_machine = json.load(f)
    else:
        with open("config/{}/changes.json".format(type_), "r") as f:
            changes_by_machine = json.load(f)

    base_config.update(base_config_2)

    if name in changes_by_machine:
        changes = changes_by_machine[name]
    else:
        changes = changes_by_machine["base"]

    # for param in changes[key]:
    #     base_config[param] = changes[key][param]

    replace_params(base_config, changes[key])

    mkpath("config/{}".format(type_))

    with open("config/{}/config_{}.json".format(type_, key), "w") as f:
        json.dump(base_config, f, indent=4)


def replace_params(base_config, changes):
    for param, value in changes.items():
        if isinstance(value, dict) and param in base_config:
            replace_params(base_config[param], changes[param])
        else:
            base_config[param] = value


def initialize_progress_bar(data_loader_list):
    num_examples = sum([len(tensor) for tensor in
                        data_loader_list.values()])
    return set_progress_bar(num_examples)


def set_progress_bar(num_examples):
    bar = tqdm(total=num_examples)
    bar.update(0)
    return bar


def merge_list_of_dicts(L):
    result = {}
    for d in L:
        result.update(d)
    return result


def return_iterator_by_type(data_type):
    if isinstance(data_type, dict):
        iterator = data_type.items()
    else:
        iterator = enumerate(data_type)
    return iterator


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def flatten(outer):
    return [el for inner in outer for el in inner]


def zipped_flatten(outer):
    return [(key, fill, el) for key, fill, inner in outer for el in inner]


def remove_none(l):
    return [e for e in l if e is not None]


# Taken from Jobman 0.1
class DD(dict):
    def __getattr__(self, attr):
        if attr == '__getstate__':
            return super(DD, self).__getstate__
        elif attr == '__setstate__':
            return super(DD, self).__setstate__
        elif attr == '__slots__':
            return super(DD, self).__slots__
        return self[attr]

    def __setattr__(self, attr, value):
        # Safety check to ensure consistent behavior with __getattr__.
        assert attr not in ('__getstate__', '__setstate__', '__slots__')
#         if attr.startswith('__'):
#             return super(DD, self).__setattr__(attr, value)
        self[attr] = value

    def __str__(self):
        return 'DD%s' % dict(self)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        z = DD()
        for k, kv in self.items():
            z[k] = copy.deepcopy(kv, memo)
        return z




# utils for hyperbole relationships and get probability values as s1, s2, s3 scores
rel_characteristic = ['CapableOf','DefinedAs','HasFirstSubevent','HasLastSubevent','HasSubevent',
    'IsA',
    'UsedFor']

causal_relations = ['CausesDesire', 'HasFirstSubevent', 'HasLastSubevent']
def getprob(input_e1, input_e2, relation, prnt = False):
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
    outputs = interactive.get_conceptnet_sequence(input_event, model, sampler, data_loader, text_encoder, relation, prnt)
    return outputs



def reverse_property(input_event, sampling_algorithm = 'beam-25'):
    reverse_sampler = reverse_interactive.set_sampler(reverse_opt, sampling_algorithm, reverse_data_loader)
    reverse_outputs = reverse_interactive.get_conceptnet_sequence(input_event, reverse_model, reverse_sampler, reverse_data_loader, reverse_text_encoder, relation = 'HasProperty')
    return reverse_outputs


def N_PN(token):
    return token.pos_ == 'NOUN' or token.pos_ =='PROPN'
def get_so_idx(doc):
    for token in doc:
        if token.lemma_ == 'so':
            return( token.i)
def parse(sentence):
    doc = nlp(sentence)
    adj = []
    so_id = get_so_idx(doc)
    NN = ''
    for token in doc:
        if N_PN(token) and token.i < so_id:
            NN += token.text
            #NN += ' '
        elif token.i > so_id and (token.pos_ == 'ADJ' or token.pos_ == 'ADV'):
            adj.append(token.text)
            NN.strip(' ')
    if NN == '':
        NN = str(doc[0])
    return NN, adj, sentence.replace(" so "," ")


def get_action(sentence):
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ =='VERB' and token.text not in ['will', 'can']:
            return str(doc[token.i:])
    return ''


def get_characteristic(inp):
    result = getPred(inp, relation=rel_characteristic, prnt = False, sampling_algorithm = 'beam-5')
    movements = []
    for rel in rel_characteristic:
        for phrase in result[rel]["beams"]:
            if get_action(phrase):
                movements.append(get_action(phrase)) 
    movements = list(set(movements))
    fit_dict={}
    for m in movements:
            p = getprob(m, subject, relation = 'RelatedTo', prnt = False)
            fit_dict[m] = p
    sorted_l = sorted(fit_dict.items(), key=operator.itemgetter(1))
    num = int(len(sorted_l)*1/2)
    good_movements = dict(sorted_l[:num]).keys()
    return good_movements

'''
# This simile model is an alternative to reverse comet model,
# That is to say, we provide another way to get objects that share certain property 
# you may well skip this simile model, as the reverse comet model is sufficien 


from fairseq.models.bart import BARTModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import time
import numpy as np
import pickle
os.environ['CUDA_VISIBLE_DEVICES']="0"

datadir = 'simile'
cpdir = path+datadir+'/'
bart = BARTModel.from_pretrained(cpdir,checkpoint_file='checkpoint_best.pt',data_name_or_path='/nas/home/yufei/fairseq/fairseq/'+datadir)

bart.cuda()
bart.eval()


maxb = 30
minb = 7
t = 0.7


def simile_vehicle(inp):
        last_word = inp.split()[-1]
        inp = inp.replace(' so ',' ')
        slines = [inp]
        answer = ''
        vehicles = []
        for seed in range(42,50):
            np.random.seed(seed)
            torch.manual_seed(seed)
            hypotheses_batch = bart.sample(slines, sampling=True, sampling_topk=5, temperature=t, lenpen=2.0, max_len_b=maxb, min_len=minb, no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                    answer = hypothesis.replace('\n','')
            vehicle = answer.strip().split(' like ')[1].replace('<EOT>','')
            if vehicle not in vehicles:
                vehicles.append(vehicle)
        return vehicles

simile_vehicle('your drawing is so bright')
'''


def hasAttribute(A, attribute, B):
    return min(getprob(A, B, 'RelatedTo'), getprob(B, A, 'RelatedTo'), getprob(B, attribute, 'HasProperty'))
def CapableOf(B,C):
    return getprob(B, C, 'CapableOf')

def characteristic(B,C):
    temp = 999
    for rel in rel_characteristic:
        temp = min(temp, getprob(B, C, rel))
    return temp


def causes(A,C):
    return min(getprob(A, C, causal_relations[0]), getprob(A, C, causal_relations[1]),
                                                           getprob(A, C, causal_relations[2]))


def get_characteristic_prob(B,C):
    temp = 999
    for rel in rel_characteristic:
        temp = min(temp, getprob(B, C, rel))
    return temp


def get_score(A, Bs, Cs, gen, all_s2, all_s3, all_s4):
    doc = nlp(A)
    so_id = get_so_idx(doc)
    attribute = doc[so_id+1:].text
    for B, C in zip(Bs, Cs):
        try:
            s2 = hasAttribute(A, attribute, B)
            full_text = A + ' even ' + B + ' '+ C +'!'
            print(full_text)
            #try:
            s3 = characteristic(B,C)
            s4 = min(getprob(A, C, causal_relations[0]), getprob(A, C, causal_relations[1]),
                                                               getprob(A, C, causal_relations[2]))
            #print(s2,s3,s4)
            gen.append(full_text)
            all_s2.append(s2)
            all_s3.append(s3)
            all_s4.append(s4)
        except:
            continue
    return gen, all_s2, all_s3, all_s4