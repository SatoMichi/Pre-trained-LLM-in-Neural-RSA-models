from pathlib import Path
import os
import numpy as np
from nltk.tokenize import word_tokenize
from functools import reduce
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from shapeworld_data import load_raw_data, get_vocab, ShapeWorld
from transformers import GPT2Tokenizer
from literal_listener_shapeworld import ShapeWorld_RNN_L0
from literal_speaker_shapeworld import S0_EncoderDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device = ",device)

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
def sentence2index(sentence):
    tokenized = tokenizer.encode(sentence)
    return tokenized

root = Path(os.path.abspath('')).parent.parent.parent.absolute()
data_path = os.path.join(root,"data\shapeworld_np")
data_list = os.listdir(data_path)

print("Generating vocab_dict ...")
vocab = get_vocab([os.path.join(data_path,d) for d in data_list])
print(vocab["w2i"])
COLOR = {"white":[1,0,0,0,0,0], "green":[0,1,0,0,0,0], "gray":[0,0,1,0,0,0], "yellow":[0,0,0,1,0,0], "red":[0,0,0,0,1,0], "blue":[0,0,0,0,0,1], "other":[0,0,0,0,0,0]}
SHAPE = {"shape":[0,0,0,0], "square":[1,0,0,0], "circle":[0,1,0,0], "rectangle":[0,0,1,0], "ellipse":[0,0,0,1]}
print("Generating Vocab_dict from GPT tokenizer ...")
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
gpt_vocab_dict = tokenizer.get_vocab()
print("Length of the GPT Vocab list is ",len(gpt_vocab_dict.keys()))
PAD = 15636
SOS= EOS = UNK = 50256
original_PAD = 0
original_SOS = 1
original_EOS = 2
original_UNK = 3
w2i = vocab["w2i"]
i2w = vocab["i2w"]

print("Preparing test dataloader ...")
d = load_raw_data(os.path.join(data_path,data_list[0]))
imgs = d["imgs"]
labels = d["labels"]
langs = d["langs"]
for i in range(1,5):
    d = load_raw_data(os.path.join(data_path,data_list[i]))
    imgs = np.vstack((imgs,d["imgs"]))
    labels = np.vstack((labels,d["labels"]))
    langs = np.hstack((langs,d["langs"]))
imgs_data_tensor = torch.tensor(imgs,dtype=torch.float)
label_data_tensor = torch.tensor(labels)
context_id_data = list(map(sentence2index,langs))
max_context_len = np.max([len(c) for c in context_id_data])
padded_context_data = torch.tensor(np.array([[SOS]+c+[EOS]+[PAD]*(max_context_len-len(c)) for c in context_id_data]))   # <sos>+context+<eos>+<pad>*
#print(imgs_data_tensor.shape, label_data_tensor.shape, padded_context_data.shape)
gpt_data = [(img,u,l) for img,l,u in zip(imgs_data_tensor,label_data_tensor,padded_context_data)]
test_split = 1000
gpt_train_data, gpt_test_data = gpt_data[:-test_split], gpt_data[-test_split:]
print("Train, Test data length = ",len(gpt_train_data),",",len(gpt_test_data))
gpt_test_batch = DataLoader(dataset=gpt_test_data,batch_size=32,shuffle=False,num_workers=0)

# Accuracy code
col_list = list(COLOR.keys())
col_list[-1] = ""
shape_list = list(SHAPE.keys())
utter_list = [" ".join([w for w in (c+" "+s).split(" ") if w]) for c in col_list for s in shape_list+[""]]
gpt_utter_list = ["".join([w for w in (c+" "+s).split(" ") if w]) for c in col_list for s in shape_list+[""]]
vocab2gpt = {g:u for u,g in zip(utter_list,gpt_utter_list)}

def decode_gpt_vocab(w):
    if w in w2i.keys(): 
        return [w2i[w]]
    elif w in vocab2gpt.keys():
        return [w2i[t] for t in vocab2gpt[w].split(" ")]
    else:
        return [original_UNK]

def gpt_lang2L0_lang(generated_langs):
    langs = [tokenizer.decode([idx for idx in generated if idx not in [PAD,SOS,EOS]]) for generated in generated_langs]
    tokens = []
    for l in langs:
        decoded = [decode_gpt_vocab(w) for w in word_tokenize(l)]+[[],[]]
        tokens.append(list(reduce(lambda x,y:x+y,decoded)))
    max_tokens_len = max([len(t) for t in tokens])
    padded_tokens = torch.tensor(np.array([[original_SOS]+ts+[original_EOS]+[original_PAD]*(max_tokens_len-len(ts)) for ts in tokens]))
    return padded_tokens

def gpt_get_l0_accuracy(speaker,literal_listener,test_batch,max_len=5):
    accs = []
    speaker.eval()
    with torch.no_grad():
        for i,(cols,lang,label) in enumerate(test_batch):
            cols, lang, label = cols.to(device), lang.to(device), label.to(device)
            generated_lang, lang_probs = speaker.generate(tokenizer,cols,label,max_len=max_len)
            output_lang = gpt_lang2L0_lang(generated_lang).to(device)
            literal_listener.eval()
            lis_labels = literal_listener(cols, output_lang)
            pred_labels = torch.argmax(lis_labels,dim=1)
            correct_labels = torch.zeros(cols.shape[0])
            acc = sum(correct_labels.to(device)==pred_labels)/len(correct_labels)
            accs.append(acc.item())
    return np.mean(accs)

def gpt_get_l1_accuracy(speaker,test_batch,max_len=5):
    accs = []
    speaker.eval()
    with torch.no_grad():
        for i,(cols,lang,label) in enumerate(test_batch):
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
            correct_labels = torch.zeros(cols.shape[0])
            acc = sum(correct_labels==pred_labels)/len(correct_labels)
            accs.append(acc.item())
    return np.mean(accs)

def train_model(speaker,criterion,optimizer,train_batch,do_break=False):
    train_loss= 0
    speaker.train()
    for cols,lang,label in train_batch:
        cols, lang, label = cols.to(device), lang.type(torch.LongTensor).to(device), label.to(device)
        optimizer.zero_grad()
        output = speaker(cols, label, lang)
        output_view = output.view(-1, output.shape[-1])
        target = lang[:,1:].reshape(-1)
        lang_loss = criterion(output_view, target)
        lang_loss.backward(retain_graph=True)
        optimizer.step()
        train_loss += lang_loss.item()
        if do_break: break
    batch_train_loss = train_loss/len(train_batch)
    return batch_train_loss

def eval_model(speaker,literal_listener,criterion,test_batch,max_len=5,do_break=False):
    test_loss = 0
    speaker.eval()
    with torch.no_grad():
        for cols,lang,label in test_batch:
            cols, lang, label = cols.to(device), lang.type(torch.LongTensor).to(device), label.to(device)
            output = speaker(cols, label, lang)
            output_view = output.view(-1, output.shape[-1])
            target = lang[:,1:].reshape(-1)
            lang_loss = criterion(output_view, target)
            test_loss += lang_loss.item()
            if do_break: break
        batch_test_loss = test_loss/len(test_batch)
        batch_test_l0_acc = gpt_get_l0_accuracy(speaker,literal_listener,test_batch,max_len=max_len)
        batch_test_l1_acc = gpt_get_l1_accuracy(speaker,test_batch,max_len=max_len)
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
        batch_train_loss = train_model(speaker,criterion,optimizer,train_batch,do_break=do_break)
        batch_test_loss,batch_test_l0_acc,batch_test_l1_acc = eval_model(speaker,literal_listener,criterion,test_batch,max_len=max_len,do_break=do_break) 
        if log:
            print("Train Loss:{:.2E}, Test Loss:{:.2E}".format(batch_train_loss,batch_test_loss))
            print("Test L0 Acc:{:.2E}, Test L1 Acc:{:.2E}".format(batch_test_l0_acc,batch_test_l1_acc))
        train_loss_list.append(batch_train_loss)
        test_loss_list.append(batch_test_loss)
        test_l0_acc_list.append(batch_test_l0_acc)
        test_l1_acc_list.append(batch_test_l1_acc)
        if batch_test_loss < best_loss:
            if log: print("Best loss saved ...")
            torch.save(speaker.to(device).state_dict(),"model_params/GPT2/shapeworld_gpt2-S0_best-loss_trainSize="+str(train_size)+"_epoch="+str(epoch)+".pth")
            best_loss = batch_test_loss
        if batch_test_l0_acc > best_l0_acc:
            if log: print("Best L0 acc saved ...")
            torch.save(speaker.to(device).state_dict(),"model_params/GPT2/shapeworld_gpt2-S0_best-l0-acc_trainSize="+str(train_size)+"_epoch="+str(epoch)+".pth")
            best_l0_acc = batch_test_l0_acc
        if batch_test_l1_acc > best_l1_acc:
            if log: print("Best L1 acc saved ...")
            torch.save(speaker.to(device).state_dict(),"model_params/GPT2/shapeworld_gpt2-S0_best-l1-acc_trainSize="+str(train_size)+"_epoch="+str(epoch)+".pth")
            best_l1_acc = batch_test_l1_acc
        if do_break: break
    return train_loss_list, test_loss_list, test_l0_acc_list, test_l1_acc_list

def run(id, epoch,train_num,log=False,do_break=False):
    literal_listener = ShapeWorld_RNN_L0(len(w2i)).to(device)
    literal_listener.load_state_dict(torch.load("model_params\shapeworld_rnn_full-data_100epoch_l0_last.pth",map_location=device))
    criterion = nn.CrossEntropyLoss()
    feat_dim = 10
    max_len = 5
    # train_batch
    print("Train data size = ",train_num)
    train_batch = DataLoader(dataset=gpt_data[:train_num],batch_size=16,shuffle=True,num_workers=0)
    # model setting
    speaker = S0_EncoderDecoder(input_size=feat_dim).to(device)
    optimizer = optim.Adam(list(speaker.parameters()),lr=0.001)
    # train and eval with epoch
    tr_loss,ts_loss,ts_l0,ts_l1 = train_and_eval_epochs(speaker,literal_listener,\
        criterion,optimizer,epoch,train_batch,gpt_test_batch,train_size=train_num,max_len=max_len,log=log,do_break=do_break)
    metrics = np.array([tr_loss,ts_loss,ts_l0,ts_l1])
    np.save("metrics/GPT2/gpt2-s0_trainSize="+str(train_num)+"_epoch="+str(epoch)+"_ID="+str(id)+".npy",metrics)

if __name__ == "__main__":
    import sys
    id = int(sys.argv[1])
    epoch = int(sys.argv[2])
    train_num = int(sys.argv[3])
    log = True if sys.argv[4]=="log" else False
    do_break = True if sys.argv[5]=="break" else False
    print("Start running Experiment ...")
    run(id, epoch, train_num, log=log, do_break=do_break)