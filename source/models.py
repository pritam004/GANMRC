import config
import transformers
import torch.nn as nn
import config_file

from transformers import  (BertConfig, BertForQuestionAnswering, BertTokenizer,XLNetConfig, XLNetForQuestionAnsweringSimple, 
XLNetTokenizer,XLMConfig, XLMForQuestionAnswering, XLMTokenizer,
RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer)

from transformers import RobertaModel

MODEL_CLASSES = {
        'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer,config_file.BERT_PATH),
        'xlnet': (XLNetConfig, XLNetForQuestionAnsweringSimple, XLNetTokenizer,config_file.XLNET_PATH),
        'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer,config_file.XLM_PATH),
        'roberta': (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer,config_file.ROBERTA_PATH),
        'distil-roberta': (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer,config_file.DISTIL_ROBERTA_PATH),
        'roberta-finetuned':(RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer,config_file.FINETUNED_ROBERTA,config_file.ROBERTA_PATH),
        'bert-finetuned':(BertConfig, BertForQuestionAnswering, BertTokenizer,config_file.FINETUNED_BERT,config_file.BERT_PATH)
    }


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.RobertaModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(1024, 3)

    def forward(self, ids, mask):
        _, o2 = self.bert(ids, attention_mask=mask)

        print(self.bert.layers)
        bo = self.bert_drop(o2)

        output = self.out(bo)
        return output

class NER_Head(nn.Module):
    def __init__(self,hidden_size,max_len):
        super(NER_Head, self).__init__()
        self.hidden_size=hidden_size
        self.max_len=max_len
        self.ner_loss=nn.CrossEntropyLoss()
        
        self.out = nn.Linear(hidden_size*max_len, 19)

    def forward(self, hid_states,ner_label):
        output=self.out(hid_states.reshape(-1,self.hidden_size*self.max_len))
        ner_loss=self.ner_loss(output,ner_label)

        return ner_loss

class Generator(nn.Module):
    def __init__(self,args):
        super(Generator,self).__init__()
        
        self.config_class, self.model_class, self.tokenizer_class,self.path_class = MODEL_CLASSES[args.model_type]
        self.model=self.model_class.from_pretrained(self.path_class)
        self.dis=nn.Linear(1024,1)
        self.dis_loss_fn=nn.BCEWithLogitsLoss()
        

    def forward(self,input_ids=None,attention_mask=None,start_positions=None,end_positions=None,is_pseudo=None):
        output=self.model(input_ids=input_ids,attention_mask=attention_mask,start_positions=start_positions,end_positions=end_positions,output_hidden_states=True)
        loss=None
        start_logits=None
        end_logits=None
        if start_positions!=None:
            loss=output['loss']

            #hiddden state of encoder
            hid_states=output['hidden_states']
            # print('hidden_state_shape',hid_states[1][:,0,:].shape)
            dis_out=self.dis(hid_states[1][:,0,:])
            dis_loss=self.dis_loss_fn(dis_out.squeeze(),is_pseudo.type_as(dis_out))
            loss+=dis_loss

        else:

            # print('hidden_state_shape',dis_out.shape)

            # print('loss ,dis_loss',loss,dis_loss)
            start_logits=output['start_logits']
            end_logits=output['end_logits']

        

        return loss,start_logits,end_logits

        


class Discriminator(nn.Module):
    def __init__(self,args):
        super(Discriminator,self).__init__()
        self.model=RobertaModel.from_pretrained(args.discriminator_path)
        self.linear=nn.Linear(1024,1)
    def forward(self,token_ids,token_mask):
        out=self.model(token_ids,token_mask)
        
        out=out['pooler_output']
        # print(out.shape,out[0])
        out=nn.Sigmoid()(self.linear(out))
        return out
class Generate(nn.Module):
    def __init__(self,args):
        super(Generate,self).__init__()
        self.config_class, self.model_class, self.tokenizer_class,self.path_class = MODEL_CLASSES[args.model_type]
        self.model=self.model_class.from_pretrained(self.path_class)
    def forward(self,input_ids=None,attention_mask=None,start_positions=None,end_positions=None,is_pseudo=None):

        # output=self.model(input_ids=input_ids,attention_mask=attention_mask,start_positions=start_positions,end_positions=end_positions)
        # start_logits=output['start_logits']
        # end_logits=output['end_logits']
        # return start_logits,end_logits
        # self.model.train()

        output=self.model(input_ids=input_ids,attention_mask=attention_mask,start_positions=start_positions,end_positions=end_positions)
        # print('inside generator',list(self.model.parameters())[0].grad)
        loss=None
        start_logits=None
        end_logits=None
        if start_positions!=None:
             loss=output['loss']

            #hiddden state of encoder
            # hid_states=output['hidden_states']
            # print('hidden_state_shape',hid_states[1][:,0,:].shape)
            # dis_out=self.dis(hid_states[1][:,0,:])
            # dis_loss=self.dis_loss_fn(dis_out.squeeze(),is_pseudo.type_as(dis_out))
            # loss+=dis_loss

        # else:

            # print('hidden_state_shape',dis_out.shape)

            # print('loss ,dis_loss',loss,dis_loss)
        start_logits=output['start_logits']
        end_logits=output['end_logits']

        return loss,start_logits,end_logits

        

        