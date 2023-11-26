import pandas as pd
import numpy as np

print("Extracting and creating files...")

filePath = './mini_codenet/data/en_mini_codenet.ftr'

df = pd.read_feather(f'{filePath}')
df = df.sample(frac=1)
# MUST NOT ADD TO 1.0??
pret_perc = 0.34
rew_perc = 0.33
# eval_perc = remainder

# split pretrain, reward model, evaluation
reward_data, pretrain_data, eval_data = np.array_split(df, [int(rew_perc*len(df)), int(pret_perc*len(df))])

# split train, val, test
train_perc = 0.8
test_perc = 0.1
# val_perc = remainder

data_dict = {}
files = ["reward", "finetune", "evaluate"]
for idx, i in enumerate([reward_data, pretrain_data, eval_data]):
    test, train, val = np.array_split(i, [int(test_perc*len(i)), int(train_perc*len(i))])
    data_dict[files[idx]] = [test, train, val]


lbls = ["test", "train", "val"]
# rew, pret, eval
for key in data_dict:
    dset = data_dict[key]
    # test, train, val
    for idx, s in enumerate(dset):
        s.reset_index(inplace=True)
        s.to_feather(f"./mini_codenet/data/split/{key}_{lbls[idx]}.ftr")
