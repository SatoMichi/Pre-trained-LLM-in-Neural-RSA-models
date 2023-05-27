from pathlib import Path
import os
from functools import reduce
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from corpus import ColorsCorpusReader
from transformers import BertTokenizer
from transformers import BertModel
from color_literal_listener import BERT_Sent_L0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device = ",device)

# Pretrained Bert word embedding model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # casedは大文字小文字区別なし
bert_model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
emb_dim = 768

def sentence2vector(sentence):
    print(sentence)
    marked_sents = "[CLS] "+sentence+" [SEP]"
    tokens = tokenizer.tokenize(marked_sents)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    bert_model.to(device)
    bert_model.eval()
    with torch.no_grad(): outputs = bert_model(tokens_tensor)
    vecs = outputs[0]
    return vecs[0],tokens

root = Path(os.path.abspath('')).parent.parent.parent.absolute()
data_path = os.path.join(root,"data")
print(data_path)
corpus = ColorsCorpusReader(os.path.join(data_path,"colors.csv"), word_count=None, normalize_colors=True)
examples = list(corpus.read())
print("Number of datapoints: {}".format(len(examples)))
colors_data = [e.get_context_data()[0] for e in examples]
utterance_data = [e.get_context_data()[1] for e in examples]

# Batching
colors_data_tensor = torch.tensor(np.array(colors_data),dtype=torch.float)
if not os.path.exists("tmp/all_contexts_embs_46994x33x768.tensor"):
    context_vecs = [c[0] for c in list(map(sentence2vector,utterance_data))]
    max_context_len = max([len(c) for c in context_vecs])
    padded_context_data = torch.vstack(tuple([torch.vstack((c.to("cpu"),torch.zeros(max_context_len-len(c),emb_dim))) for c in context_vecs]))
else:
    print("Context Padded Tensor is loaded...")
    padded_context_data = torch.load("tmp/all_contexts_embs_46994x33x768.tensor")
    padded_context_data = padded_context_data.view(-1,33,emb_dim)
print("Colors shape = ",colors_data_tensor.shape)
print("Padded context id lists shape = ",padded_context_data.shape)

sentence_vecs = torch.vstack(tuple([vecs[0] for vecs in padded_context_data]))
print("CLS vector data:, ",sentence_vecs.shape)
bert_data = [(color,torch.tensor(context,dtype=torch.float)) for color,context in zip(colors_data_tensor,sentence_vecs)]
labels = torch.zeros(len(bert_data),3)
labels[:,2] = 1.0
print("total data length = ",len(bert_data))
print("total label shape = ",labels.shape)

test_num = -1000
test_x, test_y = bert_data[test_num:], labels[test_num:]
test_dataset = list(zip(test_x,test_y))
print("test dataset length: ",len(test_dataset))
bert_test_batch = DataLoader(dataset=test_dataset,batch_size=128,shuffle=False,num_workers=0)

def train_model(model,train_batch,criterion,optimizer,do_break=False):
    train_loss = 0
    train_acc = 0
    model.train()
    #print("Start Training")
    for data,label in train_batch:
        colors, contexts = data[0].to(device), data[1].to(device)
        label = label.to(device)
        optimizer.zero_grad()
        y_pred = model(colors,contexts)
        loss = criterion(y_pred,label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_label = y_pred.argmax(1)
        correct_label = label.argmax(1)
        train_acc += (sum(pred_label==correct_label)/len(correct_label)).item()
        if do_break: break
    batch_train_loss = train_loss/len(train_batch)
    batch_train_acc = train_acc/len(train_batch)
    return batch_train_loss, batch_train_acc

def eval_model(model,test_batch,criterion,do_break=False):
    test_loss = 0
    test_acc = 0
    model.eval()
    with torch.no_grad():
        for data,label in test_batch:
            colors, contexts = data[0].to(device), data[1].to(device)
            label = label.to(device)
            y_pred = model(colors,contexts)
            test_loss += criterion(y_pred,label).item()
            pred_label = y_pred.argmax(1)
            correct_label = label.argmax(1)
            test_acc += (sum(pred_label==correct_label)/len(correct_label)).item()
            if do_break: break
    batch_test_loss = test_loss/len(test_batch)
    batch_test_acc = test_acc/len(test_batch)
    return batch_test_loss, batch_test_acc

def train_and_eval_epochs(model,criterion,optimizer,epoch,train_batch,test_batch,train_size,log=True,do_break=False):
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    best_loss = 100
    best_acc = 0
    for i in range(epoch):
        if log:
            print("##############################################")
            print("Epoch:{}/{}".format(i+1,epoch))
        batch_train_loss, batch_train_acc = train_model(model,train_batch,criterion,optimizer,do_break=do_break)
        batch_test_loss, batch_test_acc = eval_model(model,test_batch,criterion,do_break=do_break)
        if log:
            print("Train Loss:{:.2E}, Test Loss:{:.2E}".format(batch_train_loss,batch_test_loss))
            print("Train Acc:{:.2E}, Test Acc:{:.2E}".format(batch_train_acc,batch_test_acc))
        train_loss_list.append(batch_train_loss)
        test_loss_list.append(batch_test_loss)
        train_acc_list.append(batch_train_acc)
        test_acc_list.append(batch_test_acc)
        if batch_test_loss < best_loss:
            if log: print("Best Loss saved ...")
            torch.save(model.to(device).state_dict(),"model_params/BERT/bert-cls-l0_best-loss_trainSize="+str(train_size)+".pth")
            best_loss = batch_test_loss
        if batch_test_acc > best_acc:
            if log: print("Best Acc saved ...")
            torch.save(model.to(device).state_dict(),"model_params/BERT/bert-cls-l0_best-acc_trainSize="+str(train_size)+".pth")
            best_acc = batch_test_acc
        if do_break: break
    return train_loss_list,test_loss_list,train_acc_list,test_acc_list

def run(id,epoch,log=False,do_break=False):
    criterion = nn.MSELoss()
    # train size up to 80000
    for train_num in [10,50,250,1250,6250,31250]:
        # train_batch
        print("Train data size = ",train_num)
        train_x, train_y = bert_data[:train_num], labels[:train_num]
        train_dataset = list(zip(train_x,train_y))
        train_batch = DataLoader(dataset=train_dataset,batch_size=128,shuffle=True,num_workers=0)
        # model setting
        model = BERT_Sent_L0(hidden_dim=emb_dim).to(device)
        optimizer = optim.Adam(model.parameters())
        # train and eval with epoch
        tr_loss,ts_loss,tr_acc,ts_acc = train_and_eval_epochs(model,criterion,optimizer,epoch,train_batch,bert_test_batch,train_size=train_num,log=log,do_break=do_break)
        metrics = np.array([tr_loss,ts_loss,tr_acc,ts_acc])
        np.save("metrics/BERT/BERT-CLS-l0_trainSize="+str(train_num)+"_ID="+str(id)+".npy",metrics)

if __name__ == "__main__":
    import sys
    id = int(sys.argv[1])
    epoch = int(sys.argv[2])
    log = True if sys.argv[3]=="log" else False
    do_break = True if sys.argv[4]=="break" else False
    print("Start running Experiment ...")
    run(id, epoch, log=log, do_break=do_break)