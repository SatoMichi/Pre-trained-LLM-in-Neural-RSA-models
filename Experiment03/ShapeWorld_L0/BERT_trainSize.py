from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from shapeworld_data import load_raw_data, get_vocab, ShapeWorld
from literal_listener_shapeworld import ShapeWorld_BERT_Sent_L0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device = ",device)

root = Path(os.path.abspath('')).parent.parent.parent.absolute()
data_path = os.path.join(root,"data\shapeworld_np")
data_list = os.listdir(data_path)

print("Generating vocab_dict ...")
vocab = get_vocab([os.path.join(data_path,d) for d in data_list])
print(vocab["w2i"])
COLOR = {"white":[1,0,0,0,0,0], "green":[0,1,0,0,0,0], "gray":[0,0,1,0,0,0], "yellow":[0,0,0,1,0,0], "red":[0,0,0,0,1,0], "blue":[0,0,0,0,0,1], "other":[0,0,0,0,0,0]}
SHAPE = {"shape":[0,0,0,0], "square":[1,0,0,0], "circle":[0,1,0,0], "rectangle":[0,0,1,0], "ellipse":[0,0,0,1]}
original_PAD = 0
original_SOS = 1
original_EOS = 2
original_UNK = 3
w2i = vocab["w2i"]
i2w = vocab["i2w"]

print("Loading the data ...")
bert_data = []
for i in range(4):
    print(i,"th data loaded")
    d = load_raw_data(os.path.join(data_path,data_list[0]))
    bert_data += [(img,label,lang) for img,label,lang in ShapeWorld(d, vocab, bert=True, sent=True, tmp_file=str(i))]
print("Full data size: ",len(bert_data))
d = load_raw_data(os.path.join(data_path,data_list[-1]))
bert_test_batch = DataLoader(ShapeWorld(d, vocab, bert=True, sent=True, tmp_file=str(4)), batch_size=32, shuffle=False)

def get_relative_accuracy(model,test_batch):
    correct_num = 0
    total_num = 0
    for imgs,labels,langs in test_batch:
        imgs,labels,langs = imgs.to(torch.float).to(device),labels.to(torch.float).to(device),langs.to(device)
        y_pred_prob = model(imgs,langs)
        y_pred = torch.max(y_pred_prob,1)[1]
        labels = torch.max(labels,1)[1]
        correct_num += torch.sum(y_pred==labels).item()
        total_num += len(labels)
    return correct_num/total_num

def train_model(model,train_batch,criterion,optimizer,do_break=False):
    train_loss = 0
    model.train()
    #print("Start Training")
    for imgs,labels,langs in train_batch:
        imgs,labels,langs = imgs.to(torch.float).to(device),labels.to(torch.float).to(device),langs.to(device)
        optimizer.zero_grad()
        y_pred = model(imgs,langs)
        loss = criterion(y_pred,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if do_break: break
    batch_train_loss = train_loss/len(train_batch)
    batch_train_acc = get_relative_accuracy(model, train_batch)
    return batch_train_loss, batch_train_acc

def eval_model(model,test_batch,criterion,do_break=False):
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for imgs,labels,langs in test_batch:
            imgs,labels,langs = imgs.to(torch.float).to(device),labels.to(torch.float).to(device),langs.to(device)
            y_pred = model(imgs,langs)
            loss = criterion(y_pred,labels)
            test_loss += loss.item()
            if do_break: break
    batch_test_loss = test_loss/len(test_batch)
    batch_test_acc = get_relative_accuracy(model, test_batch)
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

def run(id,epoch, train_num, log=False, do_break=False):
    criterion = nn.CrossEntropyLoss()
    # train_batch
    print("Train data size = ",train_num)
    train_batch = DataLoader(dataset=bert_data[:train_num],batch_size=128,shuffle=True,num_workers=0)
    # model setting
    model = ShapeWorld_BERT_Sent_L0().to(device)
    optimizer = optim.Adam(model.parameters())
    # train and eval with epoch
    tr_loss,ts_loss,tr_acc,ts_acc = train_and_eval_epochs(model,criterion,optimizer,epoch,train_batch,bert_test_batch,train_size=train_num,log=log,do_break=do_break)
    metrics = np.array([tr_loss,ts_loss,tr_acc,ts_acc])
    np.save("metrics/BERT/bert-cls-l0_trainSize="+str(train_num)+"_ID="+str(id)+".npy",metrics)

if __name__ == "__main__":
    import sys
    id = int(sys.argv[1])
    epoch = int(sys.argv[2])
    train_num = int(sys.argv[3])
    log = True if sys.argv[4]=="log" else False
    do_break = True if sys.argv[5]=="break" else False
    print("Start running Experiment ...")
    run(id, epoch, train_num, log=log, do_break=do_break)