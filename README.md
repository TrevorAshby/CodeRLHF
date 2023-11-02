# CodeRLHF
*Fausto German\*, Will McClennen\*, Trevor Ashby\**

*\* Indicates equal contribution*

## Dataset
### Download the Dataset (optional)
To start, download the entire dataset using the command:
```
bash IBMCN.sh
```
This will download and extract the dataset (7.956GB) into: [Project_CodeNet/metadata]. ~10GB of storage will be required to perform this step.

### Variable Languages (optional)
To use variable languages, include the languages within [languages.txt] with a list of desired programming languages. For example, see [languages.txt]

```python
# List of available languages
C
C++
Python
...
...
```

[IBMCN.sh] will also perform the variable language extraction.


### Mini Subset (optional)
In this repo, we include a small sub-sample of the Dataset intended for testing purposes so that the entire dataset must not be downloaded. [./mini_codenet/data/en_mini_codenet.zip]

The subset was created using the [create_mini_codenet.py] file. This file samples a sub-set percentage of the original dataset. The default implementation for the [mini_dataset] contains every question, however only 25% of the original responses are kept.

If you want to recreate this dataset, run the following command:
```shell
python create_mini_codenet.py
```
To modify the percentage of the original dataset that [mini_dataset] is comprised of, modify the following variable on line #58 of [create_mini_codenet.py]
```python
# ...
# percentage of the original dataset that mini should consist of | MODIFY-ABLE
mini_percent = 0.25 # <-----
if mini_percent <= 1.0:
    create_mini(
        src_folder='./Project_CodeNet/',
        tgt_folder="./mini_codenet/data/",
        languages=["C++", "C", "Python"],
        mini_percent=mini_percent
    )
    print('Finished creating mini_codenet')
# ...
```

*Attention: If you use this mini subset with the original dataset, there will be overlap in example. It is recommended to use either the mini or full dataset.*


### Additional splitting (required)
By default, we automatically split the mini-codenet into 3 required sub-sections.

1. Fine-tuning subset
2. Reward model subset
3. RLHF subset

We also further split each of the 3 dataset sub-sets into train, validation, and test splits.

Use the following command to split the data into the 3 required sub-sections, and into train/test/val:
```shell
python create_mini_codenet.py
```
This file takes in 4 percentages as parameters, each representing the desired % split for each sub-section. For the 3rd split of each instance, the remainder of the two percentages is taken into account. For example, if ```train_perc = 0.8``` and ```test_perc = 0.1```, then ```val_perc = 0.1```

To modify the reward, fine-tune, and evaluation percentage splits, modify lines 11-12 in [mini_codenet_split.py]:

```python
# ...
pret_perc = 0.34 # <-------
rew_perc = 0.33 # <-------
# eval_perc = remainder

# split pretrain, reward model, evaluation
reward_data, pretrain_data, eval_data = np.array_split(df, [int(rew_perc*len(df)), int(pret_perc*len(df))])
# ...
```

To modify the train, test, and validation percentage splits, modify lines 19-20 in [mini_codenet_split.py]:

```python
# split train, val, test
train_perc = 0.8 # <-------
test_perc = 0.1 # <-------
# val_perc = remainder
```

Running [mini_codenet_split.py] will create additional feather files in [./mini_codenet/data/split/] each representing and named for the designated purpose (i.e., 'pretrain_train.ftr').