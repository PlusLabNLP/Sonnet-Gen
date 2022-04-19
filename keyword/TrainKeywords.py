#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

# define a rich console logger
console = Console(record=True)

# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)

# training logger to log training progress
training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)


# In[2]:


get_ipython().system('nvidia-smi')


# In[3]:


# Setting up the device for GPU usage
from torch import cuda
device = 'cuda:1' if cuda.is_available() else 'cpu'


# In[4]:


class YourDataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


# In[5]:


def train(epoch, tokenizer, model, device, loader, optimizer):

    """
    Function to be called for training with the parameters passed from main function

    """

    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        if _ % 1000 == 0:
            training_logger.add_row(str(epoch), str(_), str(loss))
            console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# In[6]:


def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    inputs = []
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            

            generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=512, 
                min_length = 100,
              num_beams = 4,
              no_repeat_ngram_size = 5,
              #topp = 0.9,
              #do_sample=True,
              repetition_penalty=5.8, 
              length_penalty=1, 
              early_stopping=True
              )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            input_text = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in ids]
            if _%50==0:
                outputs = model(
                        input_ids=ids,
                        attention_mask=mask,
                        decoder_input_ids=y_ids,
                        labels=lm_labels,
                        )
                loss = outputs[0]
                console.print(f'Completed {_}')
                console.print('loss: '+ str(loss))
            
            predictions.extend(preds)
            actuals.extend(target)
            inputs.extend(input_text)
    return inputs, predictions, actuals


# In[7]:


def generate(tokenizer, model, device, loader):
    model.eval()
    inputs = []
    predictions = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            

            generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=512, 
                min_length = 100,
              num_beams = 4,
              no_repeat_ngram_size = 5,
              #topp = 0.9,
              #do_sample=True,
              repetition_penalty=5.8, 
              length_penalty=1, 
              early_stopping=True
              )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            input_text = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in ids]
            if _%50==0:
                console.print(f'Completed {_}')

            predictions.extend(preds)
            inputs.extend(input_text)
    return inputs, predictions


# In[8]:


def T5Trainer(
    dataframe, source_text, target_text, model_params, model, tokenizer, output_dir="./outputs/"
):

    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
    display_df(dataframe.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size. So x% of the data will be used for training and the rest for validation.
    train_size = 0.998
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = YourDataSetClass(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    # Training loop
    console.log(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)
        console.log(f"[Initiating Validation]...\n")
        inputs, predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({'Input': inputs, "Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(os.path.join(output_dir, "predictions"+str(epoch)+".csv"))

    console.log(f"[Saving Model]...\n")
    # Saving the model after training
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    # evaluating test dataset
    

    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Validation Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")


# In[9]:


f = open('all_stories_short_theme_train.json', errors='ignore').readlines()
all_stories_short =  json.loads(f[0])
f = open('../code_news/all_news_short_theme.json', errors='ignore').readlines()
all_news_short =  json.loads(f[0])
f = open('../code_news/all_news_long_theme.json', errors='ignore').readlines()
all_news_long =  json.loads(f[0])
f = open('../code_scary_stories/all_scary_stories.json', errors='ignore').readlines()
all_scary =  json.loads(f[0])


# In[10]:


len(all_stories_short), len(all_news_short), len(all_news_long), len(all_scary)


# In[11]:


all_poetry_foundation = all_stories_short+all_news_short+all_news_long+all_scary


# In[12]:


end_token = '</s>'
X_titles = []
y_keywords = []
template = ['MASK', 'MASK', 'MASK']
prompt = 'Generate keywords for the title: '
title_set = []

for poem in all_poetry_foundation:
    title_set.append(poem['Theme'])
    title = prompt + poem['Theme']
    paddings = []
    temp = []
    count = 0
    for key in poem['keywords']:
        if key == ['<paragraph>']:
            continue
        count +=1
        mask = template[:len(key)]
        paddings.append('Keywords: '+ str(mask))        
        temp.append('Keywords: '+ str(key))
        if count >= 20:
            break

    paddings = '. '.join(paddings).replace('<paragraph> ','').replace("'","")
    temp = '. '.join(temp).replace('<paragraph> ','').replace("'","")

    X_titles.append(title + '. ' + paddings + ' '+ end_token)
    y_keywords.append(temp+ ' '+ end_token) 


# In[13]:


data = [X_titles, y_keywords]
df = pd.DataFrame(np.array(data).T, columns = ['title', 'keywords'])
df.head()


# In[14]:



# let's define model parameters specific to T5
model_params = {
    "TASK" : "keywords_mix",
    "MODEL": "t5-large",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 3,  # training batch size
    "VALID_BATCH_SIZE": 3,  # validation batch size
    "TRAIN_EPOCHS": 6,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 5e-6,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 300,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 300,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}


# In[15]:


# tokenzier for encoding the text
tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

# Further this model is sent to device (GPU/TPU) for using the hardware.
model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
model = model.to(device)


# In[19]:


output_dir= model_params['MODEL']+"_batch_size_"+ str(model_params['TRAIN_BATCH_SIZE']) + "_lr_"+ str(model_params['LEARNING_RATE'])+ "_" + model_params['TASK']
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# In[20]:


output_dir


# In[18]:


#GPU usage: 37k MB for T5 large, batch_size = 3, max_len = 550
T5Trainer(
    dataframe=df,
    source_text="title",
    target_text="keywords",
    model_params=model_params,
    model = model,
    tokenizer = tokenizer,
    output_dir = output_dir
)


# In[ ]:




