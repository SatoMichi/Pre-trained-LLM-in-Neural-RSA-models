import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Device = ",device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
emb_dim = 768

def sentence2vector(sentence):
    marked_sents = "[CLS] "+sentence+" [SEP]"
    #print(marked_sents)
    tokens = tokenizer.tokenize(marked_sents)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    bert_model.to(device)
    bert_model.eval()
    with torch.no_grad(): outputs = bert_model(tokens_tensor)
    vecs = outputs[0]
    return vecs[0],tokens

PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<UNK>'

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

class ShapeWorld:
    def __init__(self, data, vocab, bert=False, sent=False, sums=False, tmp_file=""):
        self.imgs = data['imgs']
        self.labels = data['labels']
        # Get vocab
        self.w2i = vocab['w2i']
        self.i2w = vocab['i2w']
        if len(vocab['w2i']) > 100:
            self.lang_raw = data['langs']
            self.lang_idx = data['langs']
        else:
            self.lang_raw = data['langs']
            self.lang_idx, self.lang_len = self.to_idx(self.lang_raw)
        
        self.bert = bert
        if self.bert and not os.path.exists("tmp_embs/all_contexts_embs_"+tmp_file+".tensor"):
            context_data = [txt[6:-6] for txt in self.to_text(self.lang_idx)]
            context_vecs = [c[0] for c in list(map(sentence2vector,context_data))]
            self.bert_embs = pad_sequence(context_vecs)
            torch.save(self.bert_embs,"tmp_embs/all_contexts_embs_"+tmp_file+".tensor")
            self.bert_embs = self.bert_embs.transpose(0,1)
        elif self.bert:
            self.bert_embs = torch.load("tmp_embs/all_contexts_embs_"+tmp_file+".tensor")
            self.bert_embs = self.bert_embs.transpose(0,1)
        else:
            pass
        self.sent = sent
        self.sum = sums


    def __len__(self):
        return len(self.lang_raw)

    def __getitem__(self, i):
        # Reference game format.
        img = self.imgs[i]
        label = self.labels[i]
        if self.bert and self.sent:
            #print("A")
            bert_embs_sents = torch.vstack(tuple([vecs[0] for vecs in self.bert_embs]))
            lang = bert_embs_sents[i]
        elif self.bert and self.sum:
            #print("B")
            bert_embs_sums = torch.vstack(tuple([torch.sum(vecs,dim=0) for vecs in self.bert_embs]))
            lang = bert_embs_sums[i]
        elif self.bert:
            #print("C")
            lang = self.bert_embs[i]
        else:
            #print("D")
            lang = self.lang_idx[i]
        return (img, label, lang)

    def to_text(self, idxs):
        texts = []
        for lang in idxs:
            toks = []
            for i in lang:
                i = i.item()
                if i == self.w2i[PAD_TOKEN]:
                    break
                toks.append(self.i2w.get(i, UNK_TOKEN))
            texts.append(' '.join(toks))
        return texts

    def to_idx(self, langs):
        # Add SOS, EOS
        lang_len = np.array([len(t) for t in langs], dtype=np.int) + 2
        lang_idx = np.full((len(self), max(lang_len)), self.w2i[PAD_TOKEN], dtype=np.int)
        for i, toks in enumerate(langs):
            lang_idx[i, 0] = self.w2i[SOS_TOKEN]
            for j, tok in enumerate(toks, start=1):
                lang_idx[i, j] = self.w2i.get(tok, self.w2i[UNK_TOKEN])
            lang_idx[i, j + 1] = self.w2i[EOS_TOKEN]
        return lang_idx, lang_len
    
    
class All_langs_ShapeWorld:
    def __init__(self, data, vocab, bert=False, sent=False, sums=False, tmp_file=""):
        self.imgs = data['imgs']
        self.labels = data['labels']
        # Get vocab
        self.w2i = vocab['w2i']
        self.i2w = vocab['i2w']
        if len(vocab['w2i']) > 100:
            self.lang_raw = data['langs']
            self.lang_idx = data['langs']
        else:
            self.lang_raws = [" ".join(lang).split(" # ") for lang in data['langs']]
            #print(self.lang_raws[:10])
            langs = [self.to_idx(lang_raw) for lang_raw in self.lang_raws]
            #print(langs[:10])
            self.lang_idxs, self.lang_lens = [l[0] for l in langs], [l[1] for l in langs]
            #print(self.lang_idxs.shape)
        
        self.bert = bert
        if self.bert and not os.path.exists("tmp_embs/all_langs_all_contexts_embs_"+tmp_file+".tensor"):
            context_data = [txt[6:-6] for txt in self.to_text(self.lang_idx)]
            context_vecs = [c[0] for c in list(map(sentence2vector,context_data))]
            self.bert_embs = pad_sequence(context_vecs)
            torch.save(self.bert_embs,"tmp_embs/all_langs_all_contexts_embs_"+tmp_file+".tensor")
            self.bert_embs = self.bert_embs.transpose(0,1)
        elif self.bert:
            self.bert_embs = torch.load("tmp_embs/all_langs_all_contexts_embs_"+tmp_file+".tensor")
            self.bert_embs = self.bert_embs.transpose(0,1)
        else:
            pass
        self.sent = sent
        self.sum = sums


    def __len__(self):
        return len(self.lang_raws[0])

    def __getitem__(self, i):
        # Reference game format.
        img = self.imgs[i]
        label = self.labels[i]
        if self.bert and self.sent:
            #print("A")
            bert_embs_sents = torch.vstack(tuple([vecs[0] for vecs in self.bert_embs]))
            lang = bert_embs_sents[i]
        elif self.bert and self.sum:
            #print("B")
            bert_embs_sums = torch.vstack(tuple([torch.sum(vecs,dim=0) for vecs in self.bert_embs]))
            lang = bert_embs_sums[i]
        elif self.bert:
            #print("C")
            lang = self.bert_embs[i]
        else:
            #print("D")
            print(self.lang_idxs.shape)
            lang = self.lang_idxs[i]
        return (img, label, lang)

    def to_text(self, idxs):
        texts = []
        for lang in idxs:
            toks = []
            for i in lang:
                i = i.item()
                if i == self.w2i[PAD_TOKEN]:
                    break
                toks.append(self.i2w.get(i, UNK_TOKEN))
            texts.append(' '.join(toks))
        return texts

    def to_idx(self, langs):
        # Add SOS, EOS
        lang_len = np.array([len(t) for t in langs], dtype=np.int) + 2
        lang_idx = np.full((len(self), max(lang_len)), self.w2i[PAD_TOKEN], dtype=np.int)
        for i, toks in enumerate(langs):
            lang_idx[i, 0] = self.w2i[SOS_TOKEN]
            for j, tok in enumerate(toks, start=1):
                lang_idx[i, j] = self.w2i.get(tok, self.w2i[UNK_TOKEN])
            lang_idx[i, j + 1] = self.w2i[EOS_TOKEN]
        return lang_idx, lang_len

def load_raw_data(data_file):
    data = np.load(data_file)
    # Preprocessing/tokenization
    try:
        return {
            'imgs': data['imgs'].transpose(0, 1, 4, 2, 3),
            'labels': data['labels'],
            'langs': np.array([t.lower().split() for t in data['langs']])
        }
    except:
        return {
            'imgs': data['imgs'],
            'labels': data['labels'],
            'langs': data['langs']
        }
    
def init_vocab(langs):
    i2w = {
        PAD_IDX: PAD_TOKEN,
        SOS_IDX: SOS_TOKEN,
        EOS_IDX: EOS_TOKEN,
        UNK_IDX: UNK_TOKEN,
    }
    w2i = {
        PAD_TOKEN: PAD_IDX,
        SOS_TOKEN: SOS_IDX,
        EOS_TOKEN: EOS_IDX,
        UNK_TOKEN: UNK_IDX,
    }

    for lang in langs:
        for tok in lang:
            if tok not in w2i:
                i = len(w2i)
                w2i[tok] = i
                i2w[i] = tok
    return {'w2i': w2i, 'i2w': i2w}

def get_vocab(data_list):
    langs = np.array([])
    for file in data_list:
        d = load_raw_data(file)
        langs = np.append(langs, d['langs'])
    vocab = init_vocab(langs)
    return vocab