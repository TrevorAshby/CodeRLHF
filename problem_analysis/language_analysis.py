import os
import torch
import pandas as pd
from tqdm import tqdm
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def classify_languages(path_to_files, gpu=False):
    print("Starting analysis...")
    log = open('./log.tsv', 'w')
    log.write(f"Problem\tLanguage\n")
    tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
    model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")

    if gpu:
        model.to('cuda')

    for file in tqdm(os.listdir(path_to_files)):
        file_string = open(f"{path_to_files}{file}", 'r', encoding="utf8").read()
        if gpu:
            tokens = tokenizer(file_string, return_tensors='pt', max_length=512).to('cuda')
        else:
            tokens = tokenizer(file_string, return_tensors='pt', max_length=512)
        out = model(**tokens).logits
        classification = model.config.id2label[out.argmax().item()]
        #print(f"File:{file}, Classification:{classification}")
        log.write(f"{file}\t{classification}\n")
    
    log.close()

def generate_statistics():
    df = pd.read_csv('./log.tsv', delimiter='\t')
    print("======================")
    print(df.head())
    print("======================")
    print(df.Language.value_counts())
    print("======================")

if __name__ == '__main__':
    # classify_languages('./problem_descriptions/', True) # uncomment this line if you wish to run the classification. "Specifically needs done if log.tsv is missing"
    generate_statistics()
