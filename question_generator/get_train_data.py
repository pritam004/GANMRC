import json
import pandas as pd

reddit_train=pd.read_csv('../output/reddit_ner.csv')
reddit_ques=pd.read_csv('/home/pritam/MRC/out_new/reddit_questions.csv')

df_outer = pd.merge(reddit_train, reddit_ques, on='id', how='outer')

df_outer['answer_start']=-1

for i in range(len(df_outer)):
    df_outer['answer_start'][i]=df_outer.iloc[i]['context'].find(str(df_outer.iloc[i]['answer']))

##getting data in form of squad

examples=dict()
data=[]

for i in range(len(df_outer)):
    entry=dict()
    paragraphs=[]
    paragraph=dict()

    paragraph['context']=df_outer['context'][i]

    qas=[]

    qa=dict()

    qa['id']=str(df_outer['id'][i])
    qa['question']=df_outer['question_y'][i]

    answers=[]
    answer=dict()

    answer['text']=str(df_outer['answer'][i])
    answer['answer_start']=str(df_outer['answer_start'][i])
    
    answers.append(answer)
    
    qa['answers']=answers
    qas.append(qa)
    paragraph['qas']=qas
    paragraphs.append(paragraph)
    entry['paragraphs']=paragraphs
    data.append(entry)
examples['data']=data

f=open('reddit_pseudo.json','w')
json.dump(json.dumps(examples),f)