import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

df = pd.read_feather('../mini_codenet/data/split/finetune_test.ftr')
# df.head()

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
# model.resize_token_embeddings(len(tokenizer))
# model.load_state_dict(torch.load('./model/cnds_model2.pt'))

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.3")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.3", torch_dtype=torch.float32)
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load('./model/cnds_model4.pt'))

model = model.to('cuda:0')
model.eval()

def generate(text_in, tok_in, mod_in):
    tok_text = tok_in(text_in, return_tensors='pt').to('cuda:0')
    gen_text = mod_in.generate(**tok_text)
    dec_text = tok_in.decode(gen_text[0], skip_special_tokens=True)
    return dec_text

#for i in range(10):
    #p = df.iloc[i]['problem_statement']
    #text_in = f'Python::{p}'
i = 0
prompt = df.iloc[i]['problem_statement']
print(prompt)
print('=============')
prompt = prompt.replace('\n', '')
lang = df.iloc[i]['language']
#sol = df.iloc[i]['solution']
#assist=f'{lang}\n{sol}'
formatted_prompt = (f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistantPython\n")

print(generate(formatted_prompt, tokenizer, model))
print('=============')
print(df.iloc[i]['solution'])