import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from models import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as acc



def run():
    df_train = pd.read_csv(config.TRAINING_FILE).fillna("none")
    df_valid=pd.read_csv(config.VALIDATION_FILE).fillna("none")
    df_train=df_train[:int(config.percent*len(df_train))]
    # df_valid=df_valid[:500]
    df_test=pd.read_csv(config.TESTING_FILE).fillna("none")
    # df_test=df_test[:5000]
    
    test_dataset = dataset.BERTDataset(
        hypothesis=df_test.hypothesis.values,premise=df_test.premise.values ,labels=df_test[['label']].values
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )





    train_dataset = dataset.BERTDataset(
        hypothesis=df_train['hypothesis'].values , premise=df_train['premise'].values,labels=df_train[['label']].values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.BERTDataset(
        hypothesis=df_valid['hypothesis'].values , premise=df_valid['premise'].values,labels=df_valid[['label']].values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cuda")
    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
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

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    model = nn.DataParallel(model)

    max_score = 0
 
    for epoch in range(config.EPOCHS):
        total_correct = 0.
        total_count = 0.
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        # print(outputs.shape,targets.shape)

        
        
        highest_act, pred_label = torch.max(outputs, dim=-1)
        score=acc(pred_label,targets)
        print(f"valid score  = {score} for fold {fold}")
        if score > max_score:
            torch.save(model.state_dict(), config.MODEL_PATH+str(config.percent))
            max_score = score
            
            outputs, targets = engine.eval_fn(test_data_loader, model, device)
            # outputs=nn.Softmax()(outputs)
            highest_act, pred_label = torch.max(outputs, dim=-1)
            test_score=acc(pred_label,targets)
            print(f'test_score is {test_score}')
            

           


if __name__ == "__main__":
    
    run()

