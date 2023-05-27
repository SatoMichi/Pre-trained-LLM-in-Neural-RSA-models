from pathlib import Path
import os
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from corpus import ColorsCorpusReader
from functools import reduce
import pickle
from transformers import GPT2Tokenizer
from color_literal_listener import Emb_RNN_L0
from color_literal_speaker import S0_EncoderDecoder
from tqdm import tqdm


def sentence2index(sentence):
    tokenized = tokenizer.encode(sentence)
    return tokenized


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device = ",device)

print("Generating Vocab_dict from GPT tokenizer ...")
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
gpt_vocab_dict = tokenizer.get_vocab()
with open('vocab.pkl', 'rb') as f:
    vocab_dict = pickle.load(f)
PAD = 15636
SOS= EOS = UNK = 50256
original_PAD = 0
original_SOS = 1
original_EOS = 2
original_UNK = 3

w2i = vocab_dict
i2w = {k:v for (v,k) in vocab_dict.items()}

print("Loading raw data ...")
root = Path(os.path.abspath('')).parent.parent.parent.absolute()
data_path = os.path.join(root,"data")
corpus = ColorsCorpusReader(os.path.join(data_path,"colors.csv"), word_count=None, normalize_colors=True)
examples = list(corpus.read())
colors_data = [e.get_context_data()[0] for e in examples]
utterance_data = [e.get_context_data()[1] for e in examples]

print("Constructing test data loader ...")
colors_data_tensor = torch.tensor(np.array(colors_data),dtype=torch.float)
context_id_data = list(map(sentence2index,utterance_data))
max_context_len = np.max([len(c) for c in context_id_data])
padded_context_data = torch.tensor(np.array([[SOS]+c+[EOS]+[PAD]*(max_context_len-len(c)) for c in context_id_data]))   # <sos>+context+<eos>+<pad>*
gpt_data = [(color,torch.tensor(context,dtype=torch.long)) for color,context in zip(colors_data_tensor,padded_context_data)]
label = torch.zeros(len(gpt_data),3)
label[:,2] = 1.0

test_split = 1000
test_data, test_label = gpt_data[-test_split:], label[-test_split:]
test_dataset = list(zip(test_data,test_label))
gpt_test_batch = DataLoader(dataset=test_dataset,batch_size=32,shuffle=False,num_workers=0)

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

def gpt_lang2L0_lang(generated_langs):
    langs = [tokenizer.decode([idx for idx in generated if idx not in [PAD,SOS,EOS]]) for generated in generated_langs]
    tokens = [[w2i.get(w,original_UNK) for w in word_tokenize(l)] for l in langs]
    max_tokens_len = max([len(t) for t in tokens])
    padded_tokens = torch.tensor(np.array([[original_SOS]+ts+[original_EOS]+[original_PAD]*(max_tokens_len-len(ts)) for ts in tokens]))
    return padded_tokens

def gpt_get_l0_accuracy(speaker,literal_listener,test_batch,max_len=5):
    accs = []
    speaker.eval()
    literal_listener.eval()
    with torch.no_grad():
        for i,((cols,lang),label) in enumerate(test_batch):
            cols, lang, label = cols.to(device), lang.to(device), label.to(device)
            generated_lang, lang_probs = speaker.generate(tokenizer,cols,label,max_len=max_len)
            output_lang = gpt_lang2L0_lang(generated_lang).to(device)
            lis_labels = literal_listener(cols, output_lang)
            pred_labels = get_prob_labels(lis_labels)
            correct_labels = np.zeros(cols.shape[0])+2
            acc = sum(correct_labels==pred_labels)/len(correct_labels)
            accs.append(acc)
    return np.mean(accs)

def gpt_get_l1_accuracy(speaker,test_batch,max_len=5):
    accs = []
    speaker.eval()
    for i,((cols,lang),label) in enumerate(test_batch):
        cols, lang, label = cols.to(device).to(torch.float), lang.to(device), label.to(device).to(torch.float)
        # for 1st image
        label01 = torch.zeros_like(label)
        label01[:,0] = 1.0
        generated_lang1, lang_probs1 = speaker.generate(tokenizer,cols,label01,max_len=max_len)
        # for 2nd image
        label02 = torch.zeros_like(label)
        label02[:,1] = 1.0
        generated_lang2, lang_probs2 = speaker.generate(tokenizer,cols,label02,max_len=max_len)
        # for 3rd image
        label03 = torch.zeros_like(label)
        label03[:,2] = 1.0
        generated_lang3, lang_probs3 = speaker.generate(tokenizer,cols,label03,max_len=max_len)
        # compute the probability
        prob01 = [[torch.log(word_dist[idx]+0.001).to("cpu").detach() for word_dist,idx in zip(sent,idxs)] \
                for batch,(sent,idxs) in enumerate(zip(lang_probs1,lang))]
        prob01_sums = list(map(sum,prob01))
        prob02 = [[torch.log(word_dist[idx]+0.001).to("cpu").detach() for word_dist,idx in zip(sent,idxs)] \
                for batch,(sent,idxs) in enumerate(zip(lang_probs2,lang))]
        prob02_sums = list(map(sum,prob02))
        prob03 = [[torch.log(word_dist[idx]+0.001).to("cpu").detach() for word_dist,idx in zip(sent,idxs)] \
                for batch,(sent,idxs) in enumerate(zip(lang_probs3,lang))]
        prob03_sums = list(map(sum,prob03))
        probs = F.softmax(torch.tensor(np.array([prob01_sums,prob02_sums,prob03_sums])).transpose(0,1),dim=-1)
        pred_labels = torch.argmax(probs,dim=1)
        correct_labels = torch.zeros(cols.shape[0])+2
        acc = sum(correct_labels==pred_labels)/len(correct_labels)
        accs.append(acc.item())
    return np.mean(accs)


def train_model(speaker,literal_listener,criterion,optimizer,train_batch,max_len=5,log=False,do_break=False):
    train_loss = 0
    speaker.train()
    #if log: print("Starting training ...")
    for j,((cols,lang),label) in enumerate(train_batch):
        cols, lang, label = cols.to(device), lang.to(device), label.to(device)
        optimizer.zero_grad()
        #if log: print("input to model")
        output = speaker(cols, label, lang)
        output_view = output.view(-1, output.shape[-1])
        target = lang[:,1:].reshape(-1)
        loss = criterion(output_view, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss += loss.item()
        #if log: print("computing accs")
        if do_break: break
    batch_train_loss = train_loss/len(train_batch)
    return batch_train_loss

def eval_model(speaker,literal_listener,criterion,test_batch,max_len=5,log=False,do_break=False):
    test_loss = 0
    test_l0_acc = 0
    test_l1_acc = 0
    speaker.eval()
    #if log: print("Starting evaluation ...")
    with torch.no_grad():
        for (cols,lang),label in test_batch:
            cols, lang, label = cols.to(device), lang.to(device), label.to(device)
            output = speaker(cols, label, lang)
            output_view = output.view(-1, output.shape[-1])
            target = lang[:,1:].reshape(-1)
            loss = criterion(output_view, target)
            test_loss += loss.item()
            test_l0_acc += gpt_get_l0_accuracy(speaker,literal_listener,test_batch,max_len=max_len)
            test_l1_acc += gpt_get_l1_accuracy(speaker,test_batch,max_len=max_len)
            if do_break: break
        batch_test_loss = test_loss/len(test_batch)
        batch_test_l0_acc = test_l0_acc/len(test_batch)
        batch_test_l1_acc = test_l1_acc/len(test_batch)
    return batch_test_loss, batch_test_l0_acc, batch_test_l1_acc

def train_and_eval_epoch(speaker,literal_listener,criterion,optimizer,epoch,train_batch,test_batch,train_size,max_len=5,log=True,do_break=False):
    train_loss_list = []
    test_loss_list = []
    test_l0_list = []
    test_l1_list = []
    best_loss = 100
    best_l0 = 0
    best_l1 = 0
    for i in range(epoch):
        if log:
            print("##############################################")
            print("Epoch:{}/{}".format(i+1,epoch))
        batch_train_loss = train_model(speaker,literal_listener,criterion,optimizer,train_batch,max_len=max_len,log=log,do_break=do_break)
        batch_test_loss, batch_test_l0_acc, batch_test_l1_acc = eval_model(speaker,literal_listener,criterion,test_batch,max_len=max_len,log=log,do_break=do_break)
        if log:
            print("Train Loss:{:.2E}, Test Loss:{:.2E}".format(batch_train_loss,batch_test_loss))
            print("Test L0 acc:{:.2E}".format(batch_test_l0_acc))
            print("Test L1 acc:{:.2E}".format(batch_test_l1_acc))
        train_loss_list.append(batch_train_loss)
        test_loss_list.append(batch_test_loss)
        test_l0_list.append(batch_test_l0_acc)
        test_l1_list.append(batch_test_l1_acc)
        if batch_test_loss < best_loss:
            if log: print("Best loss saved ...")
            torch.save(speaker.to(device).state_dict(),"model_params/GPT2/gpt-2-S0_best-loss_trainSize"+str(train_size)+".pth")
            best_loss = batch_test_loss
        if batch_test_l0_acc > best_l0:
            if log: print("Best L0 acc saved ...")
            torch.save(speaker.to(device).state_dict(),"model_params/GPT2/gpt-2-S0_best-l0-acc_trainSize"+str(train_size)+".pth")
            best_l0 = batch_test_l0_acc
        if batch_test_l1_acc > best_l1:
            if log: print("Best L1 acc saved ...")
            torch.save(speaker.to(device).state_dict(),"model_params/GPT2/gpt-2-S0_best-l1-acc_trainSize"+str(train_size)+".pth")
            best_l1 = batch_test_l1_acc
        if do_break: break
    return train_loss_list,test_loss_list,test_l0_list,test_l1_list

def run(id, epoch,train_num,log=False,do_break=False):
    literal_listener = Emb_RNN_L0(len(vocab_dict)).to(device)
    literal_listener.load_state_dict(torch.load("model_params\emb-rnn-l0_epoch=100_best-acc.pth",map_location=device))
    criterion = nn.CrossEntropyLoss()
    max_len = 5
    epoch = epoch
    # train_batch
    print("Train data size = ",train_num)
    train_x, train_y = gpt_data[:train_num], label[:train_num]
    train_dataset = list(zip(train_x,train_y))
    train_batch = DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,num_workers=0)
    # model setting
    speaker = S0_EncoderDecoder(input_size=3).to(device)
    optimizer = optim.Adam(list(speaker.parameters()),lr=0.001)
    # train and eval with epoch
    tr_loss,ts_loss,ts_l0,ts_l1 = train_and_eval_epoch(speaker,literal_listener,\
        criterion,optimizer,epoch,train_batch,gpt_test_batch,train_size=train_num,max_len=max_len,log=log,do_break=do_break)
    metrics = np.array([tr_loss,ts_loss,ts_l0,ts_l1])
    np.save("metrics/GPT2/gpt2-s0_trainSize="+str(train_num)+"_ID="+str(id)+".npy",metrics)

if __name__ == "__main__":
    import sys
    id  = int(sys.argv[1])
    epoch = int(sys.argv[2])
    train_num = int(sys.argv[3])
    log = True if sys.argv[4]=="log" else False
    do_break = True if sys.argv[5]=="break" else False
    print("Start running Experiment ...")
    run(id, epoch, train_num, log=log, do_break=do_break)