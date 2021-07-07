from torch.utils.data import DataLoader
from transformers import AdamW
import torch
import datasets
import utils

import config
from transformers import RobertaForQuestionAnswering
from transformers import DistilBertForQuestionAnswering
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")



train_contexts,train_questions,train_answers,val_contexts,\
    val_questions,val_answers=utils.read_data('squad')


utils.add_end_idx(train_answers, train_contexts)
utils.add_end_idx(val_answers, val_contexts)

tokenizer=config.tokenizer


train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

utils.add_token_positions(train_encodings, train_answers,tokenizer)
utils.add_token_positions(val_encodings, val_answers,tokenizer)

# model= RobertaForQuestionAnswering.from_pretrained(config.BERT_PATH)





device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

train_dataset = datasets.SquadDataset(train_encodings)
val_dataset = datasets.SquadDataset(val_encodings)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        print(loss)
        loss.backward()
        optim.step()

model.eval()