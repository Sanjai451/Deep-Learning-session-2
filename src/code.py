import requests
from bs4 import BeautifulSoup
import re
import csv

url = "https://www.wikihow.com/Special:Randomizer"

response = requests.get(url)

html_content = response.content

soup = BeautifulSoup(html_content, 'html.parser')
article_title = soup.find('title').text.strip()
print(article_title)


subheadings = []
paragraphs = []
steps = soup.find_all('div' , {'class':'step'})
for step in steps:
  subheading_elements = step.find('b')
  if(subheading_elements is not None):
    subheading_text = subheading_elements.text.strip().replace('\n','')
    subheading_text = subheading_text.encode('ascii',errors = 'ignore').decode()
    subheading_text = re.sub(r'','',subheading_text)
    print(subheading_text)
    subheadings.append(subheading_text)

    subheading_elements.extract()
    for span_tag in step.find_all('span'):
      span_tag.extract()

    paragraph_text = step.text.strip().replace('\n','').replace(' ', ' ')
    paragraph_text = paragraph_text.encode('ascii',errors = 'ignore').decode()
    paragraph_text = re.sub(r'','',paragraph_text)
    print(paragraph_text)
    paragraphs.append(paragraph_text)

if(len(subheadings)):
  with open('/content/wikiHow.csv', mode='a',newline='',encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    for i in range(len(subheadings)):
      writer.writerow([article_title,subheadings[i],paragraphs[i]])

for count in range(4000):
    url = "https://www.wikihow.com/Special:Randomizer"
    response = requests.get(url)

    html_content = response.content

    soup = BeautifulSoup(html_content, 'html.parser')
    article_title = soup.find('title').text.strip()
    print(article_title+" "+str(count))

    subheadings = []
    paragraphs = []
    steps = soup.find_all('div' , {'class':'step'})
    for step in steps:
        subheading_elements = step.find('b')
        if(subheading_elements is not None):
          subheading_text = subheading_elements.text.strip().replace('\n','')
          subheading_text = subheading_text.encode('ascii',errors = 'ignore').decode()
          subheading_text = re.sub(r'','',subheading_text)
          subheadings.append(subheading_text)
          subheading_elements.extract()
          for span_tag in step.find_all('span'):
              span_tag.extract()

          paragraph_text = step.text.strip().replace('\n','').replace(' ', ' ')
          paragraph_text = paragraph_text.encode('ascii',errors = 'ignore').decode()
          paragraph_text = re.sub(r'','',paragraph_text)
          paragraphs.append(paragraph_text)

    if(len(subheadings)):
        with open('/content/wikiHow.csv', mode='a',newline='',encoding='utf-8') as csv_file:
             writer = csv.writer(csv_file)
             for i in range(len(subheadings)):
                   writer.writerow([article_title,subheadings[i],paragraphs[i]])


# from datasets import load_metric
import pandas as pd
df = pd.read_csv("/content/wikiHow.csv")
df.head()

print(df.shape)
df = df.dropna()
print(df.shape)

print(df.shape)
df = df.drop_duplicates()
print(df.shape)

# Convert the column to string type to handle potential float values.
df['length'] = df[' paragraph'].astype(str).map(lambda x: len(x.split(" ")))


tempDf = df[df.length<=200]
tempDf.shape

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")

max_input_length = 1024
max_output_length = 64
batch_size = 16

def process_data_to_model_inputs(batch):
  inputs = tokenizer(batch[" paragraph"],
       padding="max_length",
       truncation=True,
       max_length=max_input_length
  )
  outputs = tokenizer(batch[" headings"],
       padding="max_length",
       truncation=True,
       max_length=max_output_length,
  )

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask

  batch["global_attention_mask"] = len(batch["input_ids"]) * [
      [0 for _ in range(len(batch["input_ids"][10]))]
  ]

  batch["global_attention_mask"][0] = 1
  batch["labels"] = outputs.input_ids

  batch["labels"] = [
      [-100 if token == tokenizer.pad_token_type_id else token for token in labels]
      for labels in batch["labels"]
  ]
  return batch

import numpy as np
train, validate, test = np.split(tempDf.sample(frac=1, random_state=42), [int(.6*len(df)), int(.7*len(df))])
print(train.shape)
print(validate.shape)
print(test.shape)

validate = validate[:20]
validate.shape

def process_data_to_model_inputs(batch):
       # ... your existing code ... 

       for key, value in batch.items():
           print(f"{key}: {type(value)}, {value}") 
       return batch


val_dataset = val_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["title", " headings" , " paragraph","length","__index_level_0__"],
)

from transformers import AutoModelForSeq25eqLM
led = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384", gradient_checkpointing-True, use_cache-false)
led.config.num_beans = 2
led.config.max_length = 64
led.config.min_length = 2
led.config.length_penalty = 2.0
led.config.early_stopping = True
led.config.no_repeat_ngram_size = 3
rouge =  load_metric("rouge")

def compute_metrics(pred):
    labels_ids= pred.label_ids
    pred_ids  = pred.predictions

    pred_str =  tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids = -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens-True)

    rouge_output =  rouge.compute(
       predictions - pred_str, references-label str, rouge_types ["rouge2"]
    )["rouge2"].mid
    
    return {
    "rougel_precision": round(rouge_output.precision, 4),
    "rouge2_recall": round (rouge_output.recall, 4),
    "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import transformers
transformers.logging.set_verbosity_info()

training_args =  Seq2SeqTrainingArguments(
    predict with_generate = True,
    evaluation strategy ="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size = batch size,
    outpuT_dir = "./"
    logging steps=5,
    eval steps=10,
    save steps=10,
    save_total_limit=2,
    gradient_accumulation_steps=4,
    num_train_epochs=18
)