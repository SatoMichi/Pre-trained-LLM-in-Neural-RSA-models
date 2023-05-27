import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EncoderDecoderModel
from literal_listener_shapeworld import CNN_encoder

class Imgs_emb_DeepSet(nn.Module):
    def __init__(self, input_size=10, output_size=20):
        super(Imgs_emb_DeepSet, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(input_size, output_size)
        self.linear3 = nn.Linear(output_size, output_size)
    
    def forward(self, img_emb1, img_emb2):
        img_embs = F.relu(self.linear1(img_emb1)) + F.relu(self.linear2(img_emb2))
        img_embs = self.linear3(img_embs)
        return img_embs


class CS_CNN_Encoder(nn.Module):
    def __init__(self, input_size=10, output_size=10,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(CS_CNN_Encoder, self).__init__()
        self.cnn_color_encoder = CNN_encoder(6)
        self.cnn_color_encoder.load_state_dict(torch.load("model_params/shapeworld_original-cnn_color_model.pth",map_location=device))
        for params in self.cnn_color_encoder.parameters(): params.requires_grad = False
        self.cnn_shape_encoder = CNN_encoder(4)
        self.cnn_shape_encoder.load_state_dict(torch.load("model_params/shapeworld_original-cnn_shape_model.pth",map_location=device))
        for params in self.cnn_shape_encoder.parameters(): params.requires_grad = False
        self.deepset_size = 6+4
        self.deepset = Imgs_emb_DeepSet(self.deepset_size, self.deepset_size)
        self.linear = nn.Linear(input_size+self.deepset_size, output_size)

    def get_feat_emb(self,feat):
        col_embs = self.cnn_color_encoder(feat)
        shape_embs = self.cnn_shape_encoder(feat)
        img_embs = torch.hstack((col_embs,shape_embs))
        return img_embs
    
    def forward(self,feats,labels):
        idxs = [0,1,2]
        target_idx = int(torch.argmax(labels))
        idxs.remove(target_idx)
        other_idx1,other_idx2 = idxs[0],idxs[1]
        target_img,other_img1,other_img2 = feats[:,target_idx], feats[:,other_idx1], feats[:,other_idx2]
        target_embs = self.get_feat_emb(target_img)
        other_embs1 = self.get_feat_emb(other_img1)
        other_embs2 = self.get_feat_emb(other_img2)                 # (batch_size,10)
        other_embs = F.relu(self.deepset(other_embs1,other_embs2))  # (batch_size,10)
        embs = torch.hstack((target_embs,other_embs))               # (batch_size,20)
        feat = self.linear(embs)
        return feat
    
class RNN_Speaker(nn.Module):
    def __init__(self, feat_model, embedding_module, feat_size=1024, hidden_size=100):
        super(RNN_Speaker, self).__init__()
        self.embedding = embedding_module
        self.feat_model = feat_model
        self.embedding_dim = embedding_module.embedding_dim
        self.vocab_size = embedding_module.num_embeddings
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, bidirectional=False)
        self.outputs2vocab = nn.Linear(self.hidden_size*1, self.vocab_size)                             # *2 for bidirectioanl
        self.init_h1 = nn.Linear(feat_size, self.hidden_size)
        self.init_h2 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self,feats,labels,lang,x_lens):
        feats_emb = self.feat_model(feats, labels)
        states = self.init_h2(F.relu(self.init_h1(feats_emb))).unsqueeze(0)
        #print(lang.shape)
        embedded = self.embedding(lang)
        embedded = embedded.transpose(0, 1)                               # (B,L,D) to (L,B,D)
        #print(embedded.shape)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded,x_lens.to("cpu"),enforce_sorted=False)
        packed_outputs,states = self.gru(packed_embedded, states)
        outputs, output_lens = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        outputs = self.outputs2vocab(outputs)
        predicted_onehot = F.softmax(outputs,dim=-1)
        return predicted_onehot.transpose(0, 1)

    def generate(self, feats, labels, tau=1, max_len=40, SOS=1, EOS=2):
        batch_size = feats.size(0)
        # initialize hidden states using image features
        feats_emb = self.feat_model(feats,labels)
        states = self.init_h2(F.relu(self.init_h1(feats_emb))).unsqueeze(0)
        # This contains are series of sampled onehot vectors
        lang = []
        # first input is SOS token
        inputs_onehot = torch.zeros(batch_size, self.vocab_size).to(feats.device)   # (batch_size, n_vocab)
        inputs_onehot[:, SOS] = 1.0
        inputs_onehot = inputs_onehot.unsqueeze(1)                                  # (batch_size, 1, n_vocab)
        lang.append(inputs_onehot)                                                  # Add SOS to lang
        inputs_onehot = inputs_onehot.transpose(0, 1)                               # (B,L,D) to (L,B,D)
        inputs = inputs_onehot @ self.embedding.weight                              # (1,batch_size, n_vocab) X (n_vocab, h) -> (1,batch_size, h)
        for i in range(max_len - 2):  # Have room for SOS, EOS if never sampled
            self.gru.flatten_parameters()
            outputs, states = self.gru(inputs, states)                          # outputs: (L=1,B,H)
            outputs = outputs.squeeze()                                         # outputs: (B,H)
            outputs = self.outputs2vocab(outputs)                               # outputs: (B,V)
            predicted_onehot = F.softmax(outputs, dim=-1)    # (B,V)
            lang.append(predicted_onehot.unsqueeze(1))                          # Add to lang
            inputs = (predicted_onehot.unsqueeze(0)) @ self.embedding.weight    # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
        # Add EOS if we've never sampled it
        eos_onehot = torch.zeros(batch_size, 1, self.vocab_size).to(feats.device)
        eos_onehot[:, 0, EOS] = 1.0
        lang.append(eos_onehot)
        lang_tensor = torch.cat(lang, 1)                    # (B,max_L,V)
        return lang_tensor

class S0_EncoderDecoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size=768):
        super(S0_EncoderDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = CS_CNN_Encoder(input_size, hidden_size)
        self.decoder = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "distilgpt2").decoder

    def forward(self, feats, labels, langs):
        batch_size = len(feats)
        encoder_hidden = self.encoder(feats, labels)
        decoder_hidden = encoder_hidden.reshape(batch_size,1,self.hidden_size)
        decoder_input = langs[:,:-1]
        decoder_output = self.decoder(input_ids=decoder_input, encoder_hidden_states=decoder_hidden)
        return decoder_output[0]
    
    def generate(self,tokenizer,feats,labels,max_len=5,temperature=0.7,SOS=50256):
        batch_size = len(feats)
        encoder_hidden = self.encoder(feats, labels)
        decoder_hidden = encoder_hidden.reshape(batch_size,1,self.hidden_size)
        sos = "<|endoftext|>"
        generated = torch.tensor(tokenizer.encode(sos)*batch_size).unsqueeze(1).to(decoder_hidden.device)
        probs_list = torch.zeros(batch_size,50257)
        probs_list[:,SOS] = 1.0
        probs_list = probs_list.unsqueeze(1).to(decoder_hidden.device)
        for i in range(max_len):
            #print(generated.shape)
            decoder_output = self.decoder(input_ids=generated, encoder_hidden_states=decoder_hidden)
            logits = decoder_output[0][:,-1,:]/temperature
            probs = F.softmax(logits,dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            probs_list = torch.cat((probs_list,probs.unsqueeze(1)),dim=1)
        return generated,probs_list
