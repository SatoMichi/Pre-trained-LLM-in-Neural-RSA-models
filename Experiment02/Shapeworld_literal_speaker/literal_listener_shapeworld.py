import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from vision import ConvNet

class CNN_encoder(nn.Module):
    def __init__(self,output_dim):
        super(CNN_encoder, self).__init__()
        self.output_dim = output_dim
        self.enc = ConvNet(4)
        self.fc1 = nn.Linear(1024,300)
        self.fc2 = nn.Linear(300,50)
        self.fc3 = nn.Linear(50, self.output_dim)
    
    def forward(self,img):
        x = self.enc(img)
        #print(x.shape)
        x = x.reshape(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y_prob = F.softmax(self.fc3(x),dim=1)
        return y_prob
    
 # Accuracy = 0.669
class SimpleBaseLine_ShapeWorld_L0(nn.Module):
    def __init__(self,vocab_size, emb_dim=1024) -> None:
        super(SimpleBaseLine_ShapeWorld_L0,self).__init__()
        self.embedding_dim = emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cnn_color_encoder = CNN_encoder(6)
        self.cnn_color_encoder.load_state_dict(torch.load("model_params/shapeworld_original-cnn_color_model.pth",map_location=device))
        self.cnn_shape_encoder = CNN_encoder(4)
        self.cnn_shape_encoder.load_state_dict(torch.load("model_params/shapeworld_original-cnn_shape_model.pth",map_location=device))
        self.to_hidden = nn.Linear(10,self.embedding_dim)

    def embed_features(self, imgs):
        imgs01 = imgs[:,0]
        imgs02 = imgs[:,1]
        imgs03 = imgs[:,2]
        feats_emb_flats = []
        for imgs in [imgs01,imgs02,imgs03]:
            #print(imgs.shape)
            color_embs = self.cnn_color_encoder(imgs)
            shape_embs = self.cnn_shape_encoder(imgs)
            feats_emb = torch.hstack((color_embs, shape_embs))
            feats_emb_flats.append(feats_emb)
            #print(feats_emb_flat.shape)
        cnn_emb = torch.stack(tuple(feats_emb_flats),dim=1)
        feat_embs = self.to_hidden(cnn_emb)
        #print(feat_embs.shape)
        return feat_embs

    def forward(self,imgs,contexts):
        embs = self.embedding(contexts)
        lang_embs = torch.sum(embs,dim=1).reshape(-1,self.embedding_dim)
        imgs_emb = self.embed_features(imgs)
        #print(imgs_emb.shape,lang_embs.shape)
        scores = F.softmax(torch.einsum('ijh,ih->ij', (imgs_emb, lang_embs)))
        return scores

# Accuracy = 0.814 (100 epoch)
class ShapeWorld_RNN_L0(nn.Module):
    def __init__(self,vocab_size,emb_dim=768,hidden_dim=1024) -> None:
        super(ShapeWorld_RNN_L0,self).__init__()
        self.embedding_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cnn_encoder = ConvNet(4)
        self.rnn = nn.GRU(emb_dim, hidden_dim, dropout=0.5, batch_first=True, bidirectional=True)

    def embed_features(self, feats):
        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        rest = feats.shape[2:]
        feats_flat = feats.reshape(batch_size * n_obj, *rest)
        feats_emb_flat = self.cnn_encoder(feats_flat)
        cnn_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)
        return cnn_emb

    def forward(self,imgs,contexts):
        embs = self.embedding(contexts)
        _, hidden = self.rnn(embs)
        lang_embs = hidden[-1].view(-1,self.hidden_dim)
        imgs_emb = self.embed_features(imgs)
        #print(imgs_emb.shape,lang_embs.shape)
        scores = F.softmax(torch.einsum('ijh,ih->ij', (imgs_emb, lang_embs)))
        return scores

# Accuracy = 0.510
class ShapeWorld_BERT_RNN_L0(nn.Module):
    def __init__(self,emb_dim=768,hidden_dim=1024) -> None:
        super(ShapeWorld_BERT_RNN_L0,self).__init__()
        self.embedding_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.cnn_color_encoder = CNN_encoder(6)
        self.cnn_color_encoder.load_state_dict(torch.load("model_params/shapeworld_original-cnn_color_model.pth",map_location=device))
        self.cnn_shape_encoder = CNN_encoder(4)
        self.cnn_shape_encoder.load_state_dict(torch.load("model_params/shapeworld_original-cnn_shape_model.pth",map_location=device))
        self.to_hidden = nn.Linear(10,self.hidden_dim)
        self.linear = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, dropout=0.5, batch_first=True, bidirectional=True)

    def embed_features(self, feats):
        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        rest = feats.shape[2:]
        feats_flat = feats.reshape(batch_size * n_obj, *rest)
        color_emb_flat = self.cnn_color_encoder(feats_flat)
        shape_emb_flat = self.cnn_shape_encoder(feats_flat)
        feats_emb_flat = torch.hstack((color_emb_flat, shape_emb_flat))
        cnn_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)
        feat_embs = self.to_hidden(cnn_emb)
        #print(feat_embs.shape)
        return feat_embs

    def forward(self,imgs,contexts):
        contexts = F.relu(self.linear(contexts))
        _, hidden = self.rnn(contexts)
        lang_embs = hidden[-1].view(-1,self.hidden_dim)
        imgs_emb = self.embed_features(imgs)
        #print(imgs_emb.shape,lang_embs.shape)
        scores = F.softmax(torch.einsum('ijh,ih->ij', (imgs_emb, lang_embs)))
        return scores

# Sums Accuracy = 0.604
# Sent Accuracy = 0.760 (only 10 epoch, and still have room for improvement)
class ShapeWorld_BERT_Sent_L0(nn.Module):
    def __init__(self,emb_dim=768,hidden_dim=1024) -> None:
        super(ShapeWorld_BERT_Sent_L0,self).__init__()
        self.embedding_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.liner = nn.Linear(emb_dim,hidden_dim)
        self.cnn_color_encoder = CNN_encoder(6)
        self.cnn_color_encoder.load_state_dict(torch.load("model_params/shapeworld_original-cnn_color_model.pth",map_location=device))
        self.cnn_shape_encoder = CNN_encoder(4)
        self.cnn_shape_encoder.load_state_dict(torch.load("model_params/shapeworld_original-cnn_shape_model.pth",map_location=device))
        self.to_hidden = nn.Linear(10,self.hidden_dim)

    def embed_features(self, feats):
        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        rest = feats.shape[2:]
        feats_flat = feats.reshape(batch_size * n_obj, *rest)
        color_emb_flat = self.cnn_color_encoder(feats_flat)
        shape_emb_flat = self.cnn_shape_encoder(feats_flat)
        feats_emb_flat = torch.hstack((color_emb_flat, shape_emb_flat))
        cnn_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)
        feat_embs = self.to_hidden(cnn_emb)
        #print(feat_embs.shape)
        return feat_embs

    def forward(self,imgs,contexts):
        lang_embs = F.relu(self.liner(contexts))
        imgs_emb = self.embed_features(imgs)
        #print(imgs_emb.shape,lang_embs.shape)
        scores = F.softmax(torch.einsum('ijh,ih->ij', (imgs_emb, lang_embs)))
        return scores
    
