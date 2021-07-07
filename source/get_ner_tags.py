import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from tqdm import tqdm 
import json
import sys
import pandas as pd


def read_examples(input_file):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    idx=0
    for _,entry in tqdm(enumerate(input_data)):
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            k=0
            for j in nlp(context).ents:
                idx+=1
                answer=j.text
                ner=j.label_
                examples.append(

                {
                    'context':context,
                    'answer':answer,
                    'id':idx,
                    'ner':ner,
                    'question':''
                }
                )
                k+=1
                if k>5:
                    break
    
    
    df=pd.DataFrame(examples)
    df.to_csv('../output/'+sys.argv[1]+'_ner.csv')
    

if __name__=='__main__':
    file_name=sys.argv[2]
    read_examples(file_name)