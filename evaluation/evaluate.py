import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate(text_in, tok_in, mod_in):
    tok_text = tok_in(text_in, return_tensors='pt').to('cuda:0')
    gen_text = mod_in.generate(**tok_text)
    dec_text = tok_in.decode(gen_text[0], skip_special_tokens=True)
    return dec_text

df = pd.read_feather('../mini_codenet/data/split/finetune_test.ftr')

tokenizer = AutoTokenizer.from_pretrained("../finetuning/hf_model/")
model = AutoModelForCausalLM.from_pretrained("../finetuning/hf_model/", torch_dtype=torch.float32)

model = model.to('cuda:0')
model.eval()

i = 0

des_prob_ids = set(df['problem_id'].values)
des_lang = 'Python'
des_stat = 'Accepted'

prompt = df.iloc[i]['problem_statement']

prompt = prompt.replace('\n', '')
lang = df.iloc[i]['language']

formatted_prompt = (f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistantPython\n")

generated = generate(formatted_prompt, tokenizer, model)
original = df.iloc[i]['solution']