from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datasets import load_dataset,load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import torch
nltk.download("punkt")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Model will be trained on {device}")

#loading model
#here the model can be directly selected from hugging face
# or can be downloaded on local system and can be provided the path
# Also not necesaarily 'text2text-generation' model required
# here 'text-generation' models can also be used from hugging face
# make sure to put 'text-generation'/'text2text-generation' in pipeline accordingly
model_ckpt = "any/text2text-generation/model/from/huggingface"

# to use AutoModelForCauseLM in case using Decoder-only architecture model

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model_before_training = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

#loading dataset

#here i have used .text file for dataset
# it is not mandatory
#only thing mandatory here for preparing dataset is
#that you are having required input examples and corresponding
# output examples
with open('path/to/dataset.txt','r') as file:
    lines = file.readlines()

import re
test1 = []
for line in lines:
    test1.append(re.sub('  +', ' ', line))
test = [sub.replace("'", '').replace('\n','') for sub in test1]

inputs = []
outputs = []
for item in test:
    if test.index(item)%2==0:
        # print(item)
        inputs.append(item)
    if test.index(item)%2!=0:
        # print(item,'\n')
        outputs.append(item)

dicts = {}
test_percentage = 0.9
data_index = int(test_percentage * len(inputs))
print(data_index)
dicts['prompt'] = inputs[:data_index]
dicts['code'] = outputs[:data_index]
data_train = pd.DataFrame(dicts)
data_train = data_train.sample(frac=1)
dicts = {}
dicts['prompt'] = inputs[data_index:]
dicts['code'] = outputs[data_index:]
data_val = pd.DataFrame(dicts)
data_val = data_val.sample(frac=1)

# histogram of length of dialogue and summary to fix max length 

prompt_token_length = [len(tokenizer.encode(s)) for s in data_train['prompt'] ]
code_token_length = [len(tokenizer.encode(s))for s in data_train['code']]

fig, axes = plt.subplots(1,2, figsize=(10,4))
axes[0].hist(prompt_token_length, bins=20, color='C0',edgecolor='C0')
axes[0].set_title("PROMPT Token Length")
axes[0].set_xlabel("Length")
axes[0].set_ylabel("Count")

axes[1].hist(code_token_length, bins=20, color='C0',edgecolor='C0')
axes[1].set_title("CODE Token Length")
axes[1].set_xlabel("Length")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()

def generate_batch_sized_chunks(list_of_elements, batch_size):
    for i in range(0,len(list_of_elements),batch_size):
        yield list_of_elements[i:i+batch_size]

#code for computing rouge score
def calculate_metric_on_test_ds(datasets,metric,model,tokenizer,
                                batch_size=2,device=device,column_prompt="prompt",
                                column_code="code"):
  prompt_batches = list(generate_batch_sized_chunks(datasets[column_prompt].tolist(),batch_size))
  code_batches = list(generate_batch_sized_chunks(datasets[column_code].tolist(),batch_size))
  for prompt_batch, code_batch in tqdm(
      zip(prompt_batches, code_batches), total = len(prompt_batches)):
    prompts = tokenizer(prompt_batch , max_length=256,truncation=True,
                        padding="max_length",)
# if code not working , you can always try to change hyper-paramters
# like max_length, temperature according to you or model required 
    codes = model.generate(input_ids = torch.tensor(prompts["input_ids"]).to(device),
                           attention_mask = torch.tensor(prompts["attention_mask"]).to(device),
                           num_beams=5, max_length = 256, temperature=1.1)
    decoded_codes = [tokenizer.decode(s,skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True)
    for s in codes]
    decoded_codes = [d.replace("<n>"," ") for d in decoded_codes]
    metric.add_batch(predictions = decoded_codes, references=code_batch)
  score = metric.compute()
  return score

  # here in pipeline, change 'text2text-generation' or 'text-generation' 
# according to your model choice

pipe = pipeline('text2text-generation', model = model_ckpt)

pipe_out = pipe(data_val['prompt'][5])
modelname = "name_of_your_model"
with open(f"Results_before_finetuning_{modelname}.txt",'a') as f:
    print("Input Prompt:")
    print(data_val['prompt'][5])
    print("Actual Output")
    print(data_val['code'][5])
    print("Generated Output Before Training: ")
    print(pipe_out)
rouge_names = ["rouge1","rouge2","rougeL","rougeLsum"]
rouge_metric = load_metric('rouge')
score = calculate_metric_on_test_ds(data_val,rouge_metric,model_before_training,tokenizer=tokenizer)
rouge_dict = dict((rn,score[rn].mid.fmeasure) for rn in rouge_names)
pd.DataFrame(rouge_dict, index = ['model'])


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer=tokenizer, max_input_length=128, max_target_length=128):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.inputs = data['prompt'].tolist()
        print(self.inputs[0])
        self.targets = data['code'].tolist()
        print(self.targets[0])

    def __getitem__(self, index):
      # print("inside getitem")
      input_encoding = self.tokenizer(self.inputs[index], max_length=self.max_input_length, truncation=True)
        # target_encoding = self.tokenizer(self.targets[index], max_length=self.max_target_length, truncation=True)
      with self.tokenizer.as_target_tokenizer():
          target_encoding = self.tokenizer(self.targets[index], max_length=self.max_target_length, truncation=True)

      return {
            'input_ids': input_encoding['input_ids'],
            'attention_mask': input_encoding['attention_mask'],
            'labels': target_encoding['input_ids']
        }

    def __len__(self):
        return len(self.inputs)

dataset_pt_train_class = MyDataset(data_train)
dataset_pt_eval_class = MyDataset(data_val)

from transformers import DataCollatorForSeq2Seq
import time
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer,model = model_before_training)
from transformers import TrainingArguments, Trainer
start = time.time()
trainer_args = TrainingArguments(output_dir='./result_for'+'_'+ modelname,
                                 num_train_epochs = 500,
                                 warmup_steps = 2,
                                 per_device_train_batch_size = 32,
                                 per_device_eval_batch_size = 32,
                                 weight_decay = 0.0001,
                                 logging_steps = 5,
                                 push_to_hub = False,

                                 evaluation_strategy = 'steps',
                                 eval_steps = 100,
                                 save_steps = 1e6,
                                 gradient_accumulation_steps = 16,)

trainer = Trainer(
    model = model_before_training,
    args = trainer_args,
    train_dataset=dataset_pt_train_class,
    eval_dataset = dataset_pt_eval_class,
    data_collator = seq2seq_data_collator
)

trainer.train()
trainer.save_model("./fine-tuned_"+modelname)
end = time.time()
total_time = end-start
rouge_names = ["rouge1","rouge2","rougeL","rougeLsum"]
rouge_metric = load_metric('rouge')
# score = calculate_metric_on_test_ds(data_val,rouge_metric,model_before_training,tokenizer=tokenizer)
score = calculate_metric_on_test_ds(
    data_val,rouge_metric,model = trainer.model,tokenizer = tokenizer,)
rouge_dict = dict((rn,score[rn].mid.fmeasure) for rn in rouge_names)
pd.DataFrame(rouge_dict, index = ['Fine Tuned Model'])

#testing----
#can play with this gen_kwargs to manipulate the output generation
# accordingly
gen_kwargs = {"length_penalty":0.5,"num_beams":5,"max_length":256}
del model_before_training

sample_text = data_val['prompt'][5]
# tokenizer1 = AutoTokenizer.from_pretrained(model_ckpt)
reference = data_val['code'][5]
trained_model = AutoModelForSeq2SeqLM.from_pretrained("fine-tuned_"+modelname)
pipe_from_trained = pipeline('text2text-generation', model =trained_model,tokenizer = tokenizer)

print("prompt")
print(sample_text)
print("\n Actual Code")
print(reference)
print("\n Generated Code")
print(pipe_from_trained(sample_text, **gen_kwargs))