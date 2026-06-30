from itertools import groupby
import torch
import numpy as np
from parameters import *

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, max_len = 5000):
        super().__init__()

        pe = torch.zeros(1, max_len, d_model) 
        position = torch.arange(max_len).unsqueeze(0).unsqueeze(2) 
        div_term = ( 10000.0 ** (torch.arange(0, d_model, 2)/d_model) ).unsqueeze(0).unsqueeze(0)
        pe[0,:,0::2] = torch.sin(position / div_term)
        pe[0,:,1::2] = torch.cos(position / div_term)
        self.register_buffer('pe', pe) 
                                                               
    def forward(self, x):
        x = x + self.pe[:,:x.shape[1],:]
        return x

class EncoderBlock(torch.nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout):
        super().__init__()
        
        self.MHA = torch.nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.layer_norm_1 = torch.nn.LayerNorm(d_model)
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.W1 = torch.nn.Linear(d_model,d_ff)
        self.W2 = torch.nn.Linear(d_ff,d_model)
        self.layer_norm_2 = torch.nn.LayerNorm(d_model)
        self.dropout_2 = torch.nn.Dropout(dropout)


    def forward(self, x, mask=None):
        z1, _ = self.MHA(x, x, x, mask)
        z2 = self.layer_norm_1(x + self.dropout_1(z1))
        z3 = self.W2(torch.nn.functional.relu(self.W1(z2)))		
        y = self.layer_norm_2(z2 + self.dropout_2(z3))

        return y

class Encoder(torch.nn.Module):
    def __init__(self, d_ff, transformer_layers, n_head, d_model, dropout=0.1):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.d_ff, self.transformer_layers, self.n_head, self.d_model = d_ff, transformer_layers, n_head, d_model
        for _ in range(transformer_layers):
            self.layers.append(EncoderBlock(d_model, d_ff, n_head, dropout))

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class DecoderBlock(torch.nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout):
        super().__init__()
        
        self.d_model, self.d_ff, self.n_head, self.dropout = d_model, d_ff, n_head, dropout
        
        self.MHA1 = torch.nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.layer_norm_1 = torch.nn.LayerNorm(d_model)
        self.dropout_1 = torch.nn.Dropout(dropout)
        
        # Masked
        self.MHA2 = torch.nn.MultiheadAttention(d_model, n_head,batch_first=True)
        self.layer_norm_2 = torch.nn.LayerNorm(d_model)
        self.dropout_2 = torch.nn.Dropout(dropout)
        
        self.W1 = torch.nn.Linear(d_model,d_ff)
        self.W2 = torch.nn.Linear(d_ff,d_model)
        self.layer_norm_3 = torch.nn.LayerNorm(d_model)
        self.dropout_3 = torch.nn.Dropout(dropout)
    
    def forward(self, x, enc_outputs):
        
        s = x.size()
        causal_mask = torch.triu(torch.ones(s[1], s[1], device=x.device) * float('-inf'), diagonal=1)

        X1, _ = self.MHA1(x, x, x, attn_mask=causal_mask)
        Y1 = self.layer_norm_1(x + self.dropout_1(X1))
        
        X2, _ = self.MHA2(Y1, enc_outputs, enc_outputs)
        Y2 = self.layer_norm_2(Y1 + self.dropout_2(X2))
        
        Z = self.W2(torch.nn.functional.relu(self.W1(Y2))) 
        Y3 = self.layer_norm_3(Y2 + self.dropout_2(Z))		
        return Y3, enc_outputs
    
class Decoder(torch.nn.Module):
    def __init__(self, word2ind, d_ff, transformer_layers, n_head, d_model, unkToken, padToken, dropout=0.1):
        super().__init__()
        self.word2ind = word2ind
        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]

        self.d_model, self.d_ff, self.n_head, self.tramsformer_layers = d_model, d_ff, n_head, transformer_layers

        self.layers = torch.nn.ModuleList()
        for _ in range(transformer_layers):
            self.layers.append(DecoderBlock(d_model, d_ff, n_head, dropout))
 
    def forward(self, x, state):
        for layer in self.layers:
            x, state = layer(x, state)
        
        return x, state
    
class LanguageModel(torch.nn.Module):
    def __init__(self, n_head, d_model, word2ind, unkToken, padToken, transToken, endToken):
        super().__init__()

        self.word2ind = word2ind
        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.transTokenIdx = word2ind[transToken]
        self.endTokenIdx = word2ind[endToken]
        self.transToken = transToken
        self.endToken = endToken

        self.d_model = d_model

        self.embed = torch.nn.Embedding(len(word2ind), d_model)
        self.pos_embed = PositionalEncoding(d_model)
        self.dropout = torch.nn.Dropout(0.1)

        self.encoder = Encoder(d_ff, transformer_layers, n_head, d_model)
        self.decoder = Decoder(word2ind, d_ff, transformer_layers, n_head, d_model, unkToken, padToken)

        self.projection = torch.nn.Linear(d_model, len(word2ind))

    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents_padded = [ s+(m-len(s))*[self.decoder.padTokenIdx] for s in source]
        return torch.tensor(sents_padded, dtype=torch.long, device=device)	# shape=(batch_size, seq_len)

    def save(self,fileName):
        torch.save(self.state_dict(), fileName)

    def load(self,fileName):
        self.load_state_dict(torch.load(fileName, map_location=device))
        
    def forward(self, source):

        e = []
        d = []

        for s in source:
            s_list = list(s)
            ind = s_list.index(self.transTokenIdx)
            e.append(s_list[:ind+1])
            d.append([self.transTokenIdx] + s_list[ind+1:])


        enc_tensor = self.preparePaddedBatch(e)
        dec_tensor = self.preparePaddedBatch(d)
        
        d_input = dec_tensor[:,:-1]

        e_embed = self.embed(enc_tensor)
        d_embed = self.embed(d_input)  

        encoderInput = self.dropout(self.pos_embed(e_embed))
        decoderInput = self.dropout(self.pos_embed(d_embed))

        enc_out = self.encoder(encoderInput)
        dec_out, _ = self.decoder(decoderInput, enc_out)

        Z = self.projection(dec_out).flatten(0, 1)
        Y_bar = dec_tensor[:,1:].flatten(0,1)
        
        H = torch.nn.functional.cross_entropy(Z,Y_bar,ignore_index=self.padTokenIdx, label_smoothing=0.1)
        return H
    
    @torch.no_grad()
    def generate(self, prefix, limit=100):
        
        self.eval()
        device = next(self.parameters()).device
        #generate text
        if prefix[-1] != self.transTokenIdx:

            for _ in range(limit - len(prefix)):
                X = torch.tensor([prefix], dtype=torch.long, device=device)

                E = self.embed(X)
                E = self.dropout(self.pos_embed(E))

                bs, _, d_model = E.shape
                dummy_enc = torch.zeros(bs, 1, d_model, device=device)
                dec_out, _ = self.decoder(E, dummy_enc)

                logits = self.projection(dec_out[:, -1, :])
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=1).item()
                prefix.append(next_token)

                if next_token == self.transTokenIdx or next_token == self.endTokenIdx:
                    break

            words = list(self.word2ind)
            print([words[i] for i in prefix])
            #postProcess(prefix, list(self.word2ind))
        
        # translate
        
        source_seq = torch.tensor([prefix], dtype=torch.long, device=device)
        E = self.dropout(self.pos_embed(self.embed(source_seq)))
        
        target_seq = [self.transTokenIdx]

        enc_out = self.encoder(E)

        for _ in range(limit):
            
            X = torch.tensor([target_seq], dtype=torch.long, device=device)
            E = self.dropout(self.pos_embed(self.embed(X)))

            dec_out, _ = self.decoder(E, enc_out)

            logits = self.projection(dec_out[:, -1, :])

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=1).item()

            if next_token == self.endTokenIdx or next_token == self.transTokenIdx:
                break

            target_seq.append(next_token)

        return target_seq
    

# next_tokenIdx = probs.max().item()

# if next_token == self.unkTokenIdx:
#     list_probs = probs.squeeze(0).tolist()
#     list_probs.remove(next_tokenIdx)
#     probs = torch.tensor(list_probs).unsqueeze(0)
#     next_token = torch.argmax(probs, dim=1).item()