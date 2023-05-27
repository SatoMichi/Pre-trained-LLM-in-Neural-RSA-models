from pathlib import Path
import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from shapeworld_data import load_raw_data, get_vocab, ShapeWorld
from literal_listener_shapeworld import ShapeWorld_RNN_L0
from literal_speaker_shapeworld import CS_CNN_Encoder, RNN_Speaker

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

print("Loading the dataset ...")
d = load_raw_data(os.path.join(data_path,data_list[0]))
imgs = d["imgs"]
labels = d["labels"]
langs = d["langs"]
for i in range(1,4):
    d = load_raw_data(os.path.join(data_path,data_list[i]))
    imgs = np.vstack((imgs,d["imgs"]))
    labels = np.vstack((labels,d["labels"]))
    langs = np.hstack((langs,d["langs"]))
d["imgs"] = imgs
d["labels"] = labels
d["langs"] = langs
data = [(img,label,lang) for img,label,lang in ShapeWorld(d, vocab)]


print("Prepare the test dataloader ...")
d = load_raw_data(os.path.join(data_path,data_list[-1]))
test_batch = DataLoader(ShapeWorld(d, vocab), batch_size=32, shuffle=False)

def get_prob_labels(lang_probs):
    lang_pred = []
    for probs in lang_probs:
        if probs[0]==probs[1] and probs[1]==probs[2]: # all same
            lang_pred.append(int(np.random.randint(3)))
        elif probs[0]==probs[1] and max(probs)==probs[0]:
            lang_pred.append(int(0 if np.random.randint(2)==0 else 1))
        elif probs[1]==probs[2] and max(probs)==probs[1]:
            lang_pred.append(int(1 if np.random.randint(2)==0 else 2))
        elif probs[0]==probs[2] and max(probs)==probs[1]:
            lang_pred.append(int(0 if np.random.randint(2)==0 else 2))
        else:
            lang_pred.append(int(torch.argmax(probs)))
    return np.array(lang_pred)

def get_l0_accuracy(speaker,literal_listener,test_batch,max_len=5):
    accs = []
    literal_listener.eval()
    speaker.eval()
    for i,(cols,label,lang) in enumerate(test_batch):
        cols, lang, label = cols.to(device).to(torch.float), lang.to(device), label.to(device).to(torch.float)
        lang_tensor = speaker.generate(cols, label, max_len=max_len)
        output_lang = lang_tensor.argmax(2)
        lis_labels = literal_listener(cols, output_lang)
        pred_labels = torch.argmax(lis_labels,dim=1)
        correct_labels = torch.zeros(cols.shape[0])
        acc = sum(correct_labels.to(device)==pred_labels)/len(correct_labels)
        accs.append(acc.item())
    return np.mean(accs)

def get_l1_accuracy(speaker,test_batch,max_len=5):
    accs = []
    speaker.eval()
    with torch.no_grad():
        for i,(cols,label,lang) in enumerate(test_batch):
            cols, lang, label = cols.to(device).to(torch.float), lang.to(device), label.to(device).to(torch.float)
            # for 1st image
            label01 = torch.zeros_like(label)
            label01[:,0] = 1.0
            lang_tensor1 = speaker.generate(cols, label01, max_len=max_len)
            # for 2nd image
            label02 = torch.zeros_like(label)
            label02[:,1] = 1.0
            lang_tensor2 = speaker.generate(cols, label02, max_len=max_len)
            # for 3rd image
            label03 = torch.zeros_like(label)
            label03[:,2] = 1.0
            lang_tensor3 = speaker.generate(cols, label03, max_len=max_len)
            # compute probs
            prob01 = [[torch.log(word_dist[idx]+0.001).to("cpu").detach() for word_dist,idx in zip(sent,idxs)] \
                        for batch,(sent,idxs) in enumerate(zip(lang_tensor1,lang))]
            prob01_sums = list(map(sum,prob01))
            prob02 = [[torch.log(word_dist[idx]+0.001).to("cpu").detach() for word_dist,idx in zip(sent,idxs)] \
                        for batch,(sent,idxs) in enumerate(zip(lang_tensor2,lang))]
            prob02_sums = list(map(sum,prob02))
            prob03 = [[torch.log(word_dist[idx]+0.001).to("cpu").detach() for word_dist,idx in zip(sent,idxs)] \
                        for batch,(sent,idxs) in enumerate(zip(lang_tensor3,lang))]
            prob03_sums = list(map(sum,prob03))
            probs = torch.tensor(np.array([prob01_sums,prob02_sums,prob03_sums])).transpose(0,1)
            pred_labels = get_prob_labels(probs)
            correct_labels = np.zeros(cols.shape[0])
            acc = sum(correct_labels==pred_labels)/len(correct_labels)
            accs.append(acc.item())
    return np.mean(accs)

def to_onehot(y, n):
    y_onehot = torch.zeros(y.shape[0], n).to(y.device)
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot

def train_model(speaker,criterion,optimizer,train_batch,log=False,do_break=False):
    train_loss = 0
    speaker.train()
    for cols,label,lang in train_batch:
        cols, lang, label = cols.to(device).to(torch.float), lang.to(device), label.to(device).to(torch.float)
        optimizer.zero_grad()
        x_lens = torch.tensor(np.array([3]*len(cols))).to(device)
        lang_tensor = speaker(cols, label, lang[:,:-1], x_lens=x_lens)
        output_max_len = lang_tensor.size(1)
        lang_onehot = torch.vstack(tuple([to_onehot(sent.to(torch.int64) ,len(w2i)).unsqueeze(0) for sent in lang]))
        lang_target = lang_onehot[:,1:output_max_len+1,:]
        loss = criterion(lang_tensor.reshape(-1, len(w2i)), lang_target.reshape(-1,len(w2i)))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if do_break: break
    batch_train_loss = train_loss/len(train_batch)
    return batch_train_loss

def eval_model(speaker,literal_listener,criterion,test_batch,max_len=5,log=False,do_break=False):
    test_loss = 0
    speaker.eval()
    with torch.no_grad():
        for cols,label,lang in test_batch:
            cols, lang, label = cols.to(device).to(torch.float), lang.to(device), label.to(device).to(torch.float)
            x_lens = torch.tensor(np.array([3]*len(cols))).to(device)
            lang_tensor = speaker(cols, label, lang[:,:-1], x_lens=x_lens)
            output_max_len = lang_tensor.size(1)
            lang_onehot = torch.vstack(tuple([to_onehot(sent.to(torch.int64) ,len(w2i)).unsqueeze(0) for sent in lang]))
            lang_target = lang_onehot[:,1:output_max_len+1,:]
            loss = criterion(lang_tensor.reshape(-1, len(w2i)), lang_target.reshape(-1, len(w2i)))
            test_loss += loss.item()
            if do_break: break
    batch_test_loss = test_loss/len(test_batch)
    batch_test_l0_acc = get_l0_accuracy(speaker,literal_listener,test_batch,max_len=max_len)
    batch_test_l1_acc = get_l1_accuracy(speaker,test_batch,max_len=max_len)
    return batch_test_loss, batch_test_l0_acc, batch_test_l1_acc

def train_and_eval_epochs(speaker,literal_listener,criterion,optimizer,epoch,train_batch,test_batch,train_size,max_len=5,log=False,do_break=False):
    train_loss_list = []
    test_loss_list = []
    test_l0_acc_list = []
    test_l1_acc_list = []
    best_loss = 100
    best_l0_acc = 0
    best_l1_acc = 0
    for i in range(epoch):
        if log:
            print("##############################################")
            print("Epoch:{}/{}".format(i+1,epoch))
        literal_listener.train()
        batch_train_loss = train_model(speaker,criterion,optimizer,train_batch,log=log,do_break=do_break)
        batch_test_loss,batch_test_l0_acc,batch_test_l1_acc = eval_model(speaker,literal_listener,criterion,test_batch,max_len=max_len,log=log,do_break=do_break)
        if log:
            print("Train Loss:{:.2E}, Test Loss:{:.2E}".format(batch_train_loss,batch_test_loss))
            print("Test L0 Acc:{:.2E}, Test L1 Acc:{:.2E}".format(batch_test_l0_acc,batch_test_l1_acc))
        train_loss_list.append(batch_train_loss)
        test_loss_list.append(batch_test_loss)
        test_l0_acc_list.append(batch_test_l0_acc)
        test_l1_acc_list.append(batch_test_l1_acc)
        if batch_test_loss < best_loss:
            if log: print("Best loss saved ...")
            torch.save(speaker.to(device).state_dict(),"model_params/Baseline/shapeworld_RNN-S0_best-loss_trainSize="+str(train_size)+"_epoch="+str(epoch)+".pth")
            best_loss = batch_test_loss
        if batch_test_l0_acc > best_l0_acc:
            if log: print("Best L0 acc saved ...")
            torch.save(speaker.to(device).state_dict(),"model_params/Baseline/shapeworld_RNN-S0_best-l0-acc_trainSize="+str(train_size)+"_epoch="+str(epoch)+".pth")
            best_l0_acc = batch_test_l0_acc
        if batch_test_l1_acc > best_l1_acc:
            if log: print("Best L1 acc saved ...")
            torch.save(speaker.to(device).state_dict(),"model_params/Baseline/shapeworld_RNN-S0_best-l1-acc_trainSize="+str(train_size)+"_epoch="+str(epoch)+".pth")
            best_l1_acc = batch_test_l1_acc
        if do_break: break
    return train_loss_list, test_loss_list, test_l0_acc_list, test_l1_acc_list

def run(id,epoch,log=False,do_break=False):
    literal_listener = ShapeWorld_RNN_L0(len(w2i)).to(device)
    literal_listener.load_state_dict(torch.load("model_params\shapeworld_rnn_full-data_100epoch_l0_last.pth",map_location=device))
    criterion = nn.CrossEntropyLoss()
    emb_dim = 768
    feat_dim = 100
    max_len = 5
    for train_num in [15,60,250,1000,4000]:
        # train_batch
        print("Train data size = ",train_num)
        train_batch = DataLoader(dataset=data[:train_num],batch_size=32,shuffle=True,num_workers=0)
        # model setting
        speaker_embs = nn.Embedding(len(w2i), emb_dim)
        speaker_feat = CS_CNN_Encoder(output_size=feat_dim,device=device)
        speaker = RNN_Speaker(speaker_feat, speaker_embs, feat_size=feat_dim).to(device)
        optimizer = optim.Adam(list(speaker.parameters()),lr=0.001)
        # train and eval with epoch
        tr_loss,ts_loss,ts_l0,ts_l1 = train_and_eval_epochs(speaker,literal_listener,\
            criterion,optimizer,epoch,train_batch,test_batch,train_size=train_num,max_len=max_len,log=log,do_break=do_break)
        metrics = np.array([tr_loss,ts_loss,ts_l0,ts_l1])
        np.save("metrics/Baseline/baseline-s0_trainSize="+str(train_num)+"_epoch="+str(epoch)+"_ID="+str(id)+".npy",metrics)

if __name__ == "__main__":
    import sys
    id = int(sys.argv[1])
    epoch = int(sys.argv[2])
    log = True if sys.argv[3]=="log" else False
    do_break = True if sys.argv[4]=="break" else False
    print("Start running Experiment ...")
    run(id, epoch, log=log, do_break=do_break)