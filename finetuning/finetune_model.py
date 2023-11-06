import json
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

def train(model, dl, tokenizer, device, num_epochs=10):
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
            prob, sol = batch #! FOR DETAIL ON THESE SEE LINE 78
            # initialize calculated gradients (from prev step)
            criteria.zero_grad()
            # pull all tensor batches required for training

            inputs = prob.to(device)
            # attention_mask = batch['attention_mask'].to(device)
            labels = sol.to(device)
            # process
            outputs = model(
                input_ids=inputs.squeeze(1),
                labels=labels.squeeze(1),
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


# our dataset class
class CodeNetDataset(Dataset):
    def __init__(self, feather_path, tokenizer, max_len):
        self.data = pd.read_feather(feather_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data.iloc[idx]
        # language = self.tokenizer(instance["language"], return_tensors='pt')
        problem = self.tokenizer(instance["language"] + "::" + instance["problem_statement"], max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt').input_ids
        code_solution = self.tokenizer(instance["solution"], max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt').input_ids
        # encoding = self.tokenizer(instance["language"] + "::" + instance["problem_statement"] + "::" + instance["solution"], max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return problem, code_solution
        # return encoding

device_id = '0'
max_length = 512
batch_size = 1
epoch_num = 2    

if __name__ == '__main__':
    # set the device
    device = torch.device('cuda')
    # device = torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained("gpt2-large")
    model.resize_token_embeddings(len(tokenizer))

    cnds = CodeNetDataset('../mini_codenet/data/split/pretrain_train.ftr', tokenizer, max_length)
    dl = DataLoader(cnds, batch_size=batch_size, shuffle=True)
    # print(model.config)

    # load models to GPU
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=[0])

    train(model, dl, tokenizer, device, epoch_num)

    try:
        torch.save(model.module.state_dict(), './model/cnds_model.pt')
    except AttributeError:
        torch.save(model.state_dict(), './model/cnds_model.pt')

    
    