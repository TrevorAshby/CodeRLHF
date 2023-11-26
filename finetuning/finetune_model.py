import json
import os
import datetime
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader 
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

# THIS FILE IS USED TO TRAIN OUR RESPONSE GENERATOR

def train(model, dl, tokenizer, device, num_epochs=10, num_mods=1):
    model.train()
    print('Starting model training...')

    log = open('./tuning_log.txt', 'a+')
    log.write('================ STARTING A NEW RUN == {} =================\n'.format(datetime.datetime.now()))

    criteria = AdamW(model.parameters(), lr=1e-6)

    for epoch in range(num_epochs):
        eploss = 0

        # setup loop with TQDM and dataloader
        loop = tqdm(dl, leave=True)
        for batch in loop:
            #prob, sol = batch #! FOR DETAIL ON THESE SEE LINE 78
            # initialize calculated gradients (from prev step)
            criteria.zero_grad()
            # pull all tensor batches required for training

            inputs = batch.to(device)
            # print(inputs.shape)
            # attention_mask = batch['attention_mask'].to(device)
            #labels = sol.to(device)
            # process
            outputs = model(
                input_ids=inputs.squeeze(1),
                labels=inputs.squeeze(1),
            )

            # extract loss
            loss = outputs.loss
            # # calculate loss for every parameter that needs grad update
            loss = loss.mean()
            
            loss.backward()
            # update parameters
            criteria.step()

            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

            eploss += loss.item()

        log.write("Epoch:{}, EpLoss1:{}\n".format(epoch, eploss/len(dl)))
        print("Epoch:{}, EpLoss1:{}\n".format(epoch, eploss/len(dl)))

        try:
            torch.save(model.module.state_dict(), f'./model/cnds_model{num_mods}.pt')

            model.module.save_pretrained('./hf_model/')
            tokenizer.save_pretrained('./hf_model/')
        except AttributeError:
            torch.save(model.state_dict(), f'./model/cnds_model{num_mods}.pt')

            model.module.save_pretrained('./hf_model/')
            tokenizer.save_pretrained('./hf_model/')


# our dataset class
class CodeNetDataset(Dataset):
    def __init__(self, feather_path, tokenizer, max_len):#_in, max_len_out):
        self.data = pd.read_feather(feather_path)
        self.tokenizer = tokenizer
        #self.max_len_in = max_len_in
        #self.max_len_out = max_len_out
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data.iloc[idx]
        # language = self.tokenizer(instance["language"], return_tensors='pt')
        #problem = self.tokenizer(instance["language"] + "::" + instance["problem_statement"], max_length=self.max_len_in, padding='max_length', truncation=True, return_tensors='pt').input_ids
        #code_solution = self.tokenizer(instance["language"] + "::" + instance["problem_statement"] + "::" + instance["solution"], max_length=self.max_len_out, padding='max_length', truncation=True, return_tensors='pt').input_ids
        prompt = instance['problem_statement']
        prompt = prompt.replace('\n', '')
        lang = instance['language']
        sol = instance['solution']
        assist=f'{lang}\n{sol}'
        formatted_prompt = (f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant{assist}<|im_end|>\n")
        enc = self.tokenizer(formatted_prompt, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt').input_ids
        
        # encoding = self.tokenizer(instance["language"] + "::" + instance["problem_statement"] + "::" + instance["solution"], max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        # return problem, code_solution
        return enc

device_id = '0'
max_length_in = 1024 # for final 1024
# max_length_out = 1024 # for final # 2048
batch_size = 8
epoch_num = 4

if __name__ == '__main__':
    # set the device
    device = torch.device('cuda:0')
    # device = torch.device('cpu')

    tokenid = open('./token.txt', 'r').read()

    # ---- NEO ----
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", torch_dtype=torch.float32)
    # model.resize_token_embeddings(len(tokenizer))
    

    # ---- LLAMA2 ----
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=tokenid)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float32, token=tokenid)
    # model.resize_token_embeddings(len(tokenizer))

    # ---- TINYLLAMA ----
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.3")
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.3", torch_dtype=torch.float32)
    # model.load_state_dict(torch.load('./model/cnds_model3.pt'))
    # model.resize_token_embeddings(len(tokenizer))

    # ---- GPT ----
    # tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model = AutoModelForCausalLM.from_pretrained("gpt2-large")
    # model.resize_token_embeddings(len(tokenizer))

    cnds = CodeNetDataset('../mini_codenet/data/split/finetune_train.ftr', tokenizer, max_length_in)
    dl = DataLoader(cnds, batch_size=batch_size, shuffle=True)
    # print(model.config)

    # load models to GPU
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    # model.load_state_dict(torch.load('./model/cnds_model.pt'))

    num_mods = len(os.listdir('./model/'))

    train(model, dl, tokenizer, device, epoch_num, num_mods)

    
    