
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from adversarial_question_model import Discriminator
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as acc

import argparse
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from tqdm import tqdm



class BERTDataset:
    def __init__(self, question,label,answers,tokenizer,max_len):
        self.question = question
        self.answers=answers
        self.label=label
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.question)

    def __getitem__(self, item):
        question = str(self.question[item])
        question = " ".join(question.split())
        answers = str(self.answers[item])
        answers = " ".join(answers.split())
        
        


        inputs = self.tokenizer.encode_plus(
            question,
            
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
      

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
           
            "labels": torch.tensor(self.label[item], dtype=torch.long),
        }

def loss_fn(outputs, targets):

   
    return nn.BCEWithLogitsLoss()(outputs,targets)

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        #token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        #print(mask)
        targets = d["labels"]

        ids = ids.to(device, dtype=torch.long)
        #token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
   
        optimizer.zero_grad()
        outputs = model(token_ids=ids, token_masks=mask)
      #  print(targets.shape,outputs.shape)

        loss = loss_fn(outputs, targets)
        # print(loss)

       # print('i came here')
        loss.backward()
        optimizer.step()
        scheduler.step()
        # return model


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    total_correct = 0.
    total_count = 0.
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            #token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["labels"]

            ids = ids.to(device, dtype=torch.long)
            #token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(token_ids=ids, token_masks=mask)

            fin_targets.extend(targets.detach().cpu().numpy())
            fin_outputs.extend(outputs.detach().cpu().numpy())
        fin_targets=torch.tensor(fin_targets, dtype=torch.float)
        fin_outputs=torch.tensor(fin_outputs, dtype=torch.float)
    return fin_outputs, fin_targets



def run(args):
    df_squad = pd.read_csv(args.squad_file).fillna("none")
    df_target=pd.read_csv(args.target_file).fillna("none")
    df_target_ner=pd.read_csv(args.target_file_ner).fillna("none").drop('question',axis=1)
    
    df_squad=df_squad[:len(df_target)]
    df_target=df_target.join(df_target_ner,on='id',how='left',lsuffix='_left', rsuffix='_right')

    true_questions=df_squad['question'].values
    fake_questions=df_target['question'].values

    true_answers=df_squad['answer'].values
    fake_answers=df_target['answer'].values

    questions=np.concatenate((true_questions,fake_questions),0)
    answers=np.concatenate((true_answers,fake_answers),0)

    labels=np.array([1 for i in range(len(true_questions))]+[0 for i in range (len(fake_questions))])

    train_questions,test_questions,train_labels,test_labels,train_answers,test_answers=tts(questions,labels,answers,test_size=.3)

    
    TOKENIZER = RobertaTokenizer.from_pretrained(args.model_path, do_lower_case=True)
    config=config = RobertaConfig.from_pretrained(args.model_path)
    discriminator=RobertaModel.from_pretrained(args.model_path,config=config)

    network_discriminator=Discriminator(discriminator,config)


    train_dataset = BERTDataset(
        question=train_questions ,label=np.expand_dims(train_labels,1),answers=train_answers,tokenizer=TOKENIZER,max_len=args.max_len
    )
    valid_dataset = BERTDataset(
        question=test_questions ,label=np.expand_dims(test_labels,1),answers=test_answers,tokenizer=TOKENIZER,max_len=args.max_len
    )


    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, num_workers=4
    )

   

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.valid_batch_size, num_workers=1
    )

    device = torch.device("cuda")
    # model = network_discriminator
    network_discriminator.to(device)

    param_optimizer = list(network_discriminator.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_questions) / args.train_batch_size* 20)
    optimizer = AdamW(optimizer_parameters, lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    network_discriminator = nn.DataParallel(network_discriminator)

    max_score = 0
 
    for epoch in range(20):
        total_correct = 0.
        total_count = 0.
        train_fn(train_data_loader, network_discriminator, optimizer, device, scheduler)
        outputs, targets = eval_fn(valid_data_loader, network_discriminator, device)
        # print(outputs.shape,targets.shape)
        # print(outputs[:10],targets)

        
        
        # highest_act, pred_label = torch.max(outputs, dim=-1)
        
        out=[]
        for i in outputs:
            if i>0.5:
                out.append(1)
            else:
                out.append(0)
        score=acc(out,targets)

        print(f"valid score  = {score}")
        if score > max_score:
            torch.save(network_discriminator.state_dict(), args.out_dir)
            max_score = score
            
            
            

           


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default='roberta', type=str, 
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_path", default='/home/pritam/MRC/pretrained_model/roberta', type=str, 
                        help="Path to pre-trained model: e.g. roberta-base" )  
    parser.add_argument("--train_batch_size", default=32, type=int, 
                        help="Model type: e.g. roberta")
    parser.add_argument("--valid_batch_size", default=32, type=int, 
                        help="Model type: e.g. roberta")
    parser.add_argument("--out_dir", default='fine_dis/model.bin', type=str, 
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--max_len", default=128, type=int, 
                        help="Model type: e.g. roberta")
    parser.add_argument("--squad_file", default='/home/pritam/MRC/data/train_squad.csv', type=str, 
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--target_file", default='/home/pritam/MRC/out_new/reddit_questions.csv', type=str, 
                        help="Path to pre-trained model: e.g. roberta-base" ) 
    parser.add_argument("--target_file_ner", default='/home/pritam/MRC/output/reddit_ner.csv', type=str, 
                        help="Path to pre-trained model: e.g. roberta-base" ) 
    args = parser.parse_args()

    
    run(args)

