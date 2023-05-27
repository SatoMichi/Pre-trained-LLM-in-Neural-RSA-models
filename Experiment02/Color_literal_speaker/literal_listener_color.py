import torch
import torch.nn as nn
import torch.nn.functional as F
    
class Emb_RNN_L0(nn.Module):
    def __init__(self, vocab_size, emb_dim=768, hidden_dim=100, output_dim=1):
        super(Emb_RNN_L0,self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.GRU(emb_dim, hidden_dim, dropout=0.5,batch_first=True, bidirectional=True)
        self.linear01 = nn.Linear(3+hidden_dim,hidden_dim)
        self.linear02 = nn.Linear(hidden_dim,hidden_dim//2)
        self.linear03 = nn.Linear(hidden_dim//2, output_dim)

    def forward(self, color_rgbs, context):
        embs = self.embedding(context)
        output, hidden = self.lstm(embs)
        hidden = hidden[-1].view(-1,self.hidden_dim)
        y1 = self.linear03(F.relu(self.linear02(F.relu(self.linear01(torch.hstack((color_rgbs[:,0],hidden)))))))
        y2 = self.linear03(F.relu(self.linear02(F.relu(self.linear01(torch.hstack((color_rgbs[:,1],hidden)))))))
        y3 = self.linear03(F.relu(self.linear02(F.relu(self.linear01(torch.hstack((color_rgbs[:,2],hidden)))))))
        y_hat = F.softmax(torch.cat([y1,y2,y3],dim=1),dim=-1)
        return y_hat