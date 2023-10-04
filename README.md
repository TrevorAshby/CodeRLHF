# CodeRLHF
*Fausto German\*, Will McClennen\*, Trevor Ashby\**

*\* Indicates equal contribution*

## Dataset
### Download the Dataset
To start, download the entire dataset using the command:
```
bash IBMCN.sh
```
This will download and extract the dataset (7.956GB) into: [Project_CodeNet/metadata]. ~10GB of storage will be required to perform this step.

### Variable Languages
To use variable languages, include the languages within [languages.txt] with a list of desired programming languages. For example, see [languages.txt]

```python
# List of available languages
C
C++
Python
...
...
```

[IBMCN.sh] will also perform the variable language extraction. After this is complete, delete Project_CodeNet/metadata

### Train, Validation, Test split
By default, determined through testing, we automatically split the original dataset into 3 required sub-sections.

1. Pre-Training subset
2. Reward model subset
3. RLHF subset

To further split each of the 3 dataset sub-sets into train, validation, and test splits, use the following command:
```shell
python xxxx.xxx
```
This file takes in 3 percentages as parameters, each representing the desired % split for each sub-section.

To modify the x, y, and z percentage splits, modify lines x-y in [xxxx.xxx]:

```python
# ...
x = 0.70 # percentage of train set | MODIFY-ABLE
y = 0.15 # percentage of validation set | MODIFY-ABLE
z = 0.15 # percentage of test set | MODIFY-ABLE
dataset_path = './mini_codenet/data/' # path to dataset that needs to be split | MODIFY-ABLE


if 1.0 == (x + y + z): # make sure that all percentages sum to 1.0 (representing 100%)

# ...
```

### Mini Subset
In this repo, we include a small sub-sample of the Dataset intended for testing purposes so that the entire dataset must not be downloaded.

The subset was created using the [create_mini_codenet.py] file. This file separates the data into a x% train, y% validation, z% test split, while containg 3% of the original dataset. The [mini_dataset] contains every question, however only 3% of the original responses.

If you want to recreate this dataset, run the following command:
```shell
python create_mini_codenet.py
```
To modify the percentage of the original dataset that [mini_dataset] is comprised of, modify the following variable on line #26 of [create_mini_codenet.py]
```python
# ...
mini_percent = 0.1 # percentage of the original dataset that mini should consist of | MODIFY-ABLE
if mini_percent <= 1.0:
    create_mini('./Project_CodeNet/data/', mini_percent)
    print('Finished creating mini_codenet')
else:
    print('ERROR: MAKE SURE THAT mini_percent IS <= 1.0')
# ...
```

*Attention: If you use this mini subset with the original dataset, there will be overlap in example. It is recommended to use either the mini or full dataset.*