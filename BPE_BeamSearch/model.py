from itertools import groupby
import pickle
import torch
import numpy as np
from parameters import *
from utils import postProcess

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
    def generate(self, prefix, k=3, limit=150):
        self.eval()
        device = next(self.parameters()).device
        vocab_size = len(self.word2ind) 
        
        source_seq = torch.tensor([prefix], dtype=torch.long, device=device)
        E = self.dropout(self.pos_embed(self.embed(source_seq)))
        enc_out = self.encoder(E) 
        
        seqs = torch.tensor([[self.transTokenIdx]], dtype=torch.long, device=device)
        scores = torch.tensor([0.0], device=device)
        
        completed_hypotheses = []

        for step in range(limit):

            curr_batch_size = seqs.size(0)
            enc_out_batched = enc_out.expand(curr_batch_size, -1, -1)
            E_dec = self.dropout(self.pos_embed(self.embed(seqs)))

            dec_out, _ = self.decoder(E_dec, enc_out_batched)
            
            logits = self.projection(dec_out[:, -1, :]) 

            repetition_penalty = 1.2 # Values between 1.1 and 1.5 work best
            for batch_idx in range(curr_batch_size):
                for token_idx in set(seqs[batch_idx].tolist()):
                    # Penalize the logit score for tokens already generated
                    if logits[batch_idx, token_idx] < 0:
                        logits[batch_idx, token_idx] *= repetition_penalty
                    else:
                        logits[batch_idx, token_idx] /= repetition_penalty

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            next_scores = scores.unsqueeze(1) + log_probs 
            
            next_scores_flat = next_scores.view(-1)
            
            num_candidates = min(k, next_scores_flat.size(0))
            topk_scores, topk_indices = torch.topk(next_scores_flat, num_candidates)
            
            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size
            
            new_seqs = []
            new_scores = []
            
            for i in range(num_candidates):
                beam_idx = beam_indices[i].item()
                token_idx = token_indices[i].item()
                score = topk_scores[i].item()
                
                seq = seqs[beam_idx].tolist() + [token_idx]
                
                if token_idx == self.endTokenIdx:# or token_idx == self.transTokenIdx:
                    if step > 0: 
                        completed_hypotheses.append((seq, score))
                else:
                    new_seqs.append(seq)
                    new_scores.append(score)
            
            if len(completed_hypotheses) >= k:
                break
                
            if not new_seqs:
                break
                
            active_k = k - len(completed_hypotheses)
            new_seqs = new_seqs[:active_k]
            new_scores = new_scores[:active_k]
            
            seqs = torch.tensor(new_seqs, dtype=torch.long, device=device)
            scores = torch.tensor(new_scores, device=device)

        for i in range(len(new_seqs)):
            completed_hypotheses.append((new_seqs[i], new_scores[i]))
            
        ordered = sorted(completed_hypotheses, key=lambda x: x[1], reverse=True)
        
        #return ordered[:k]
        return ordered[0][0]





    # @torch.no_grad()
    # def generate(self, prefix, k=300, limit=1000):

    #     self.eval()
    #     device = next(self.parameters()).device
        
    #     source_seq = torch.tensor([prefix], dtype=torch.long, device=device)
    #     E = self.dropout(self.pos_embed(self.embed(source_seq)))
    #     enc_out = self.encoder(E)
        
    #     beam = [([self.transTokenIdx], 0.0)]
        
    #     for step in range(limit):
    #         all_candidates = []
            
    #         for seq, score in beam:
                  
    #             if seq[-1] == self.endTokenIdx or seq[-1] == self.transTokenIdx:
    #                 if step > 0:
    #                     all_candidates.append((seq, score)) 
    #                     continue
                
    #             X = torch.tensor([seq], dtype=torch.long, device=device)
    #             E_dec = self.dropout(self.pos_embed(self.embed(X)))

    #             dec_out, _ = self.decoder(E_dec, enc_out)
                
    #             logits = self.projection(dec_out[:, -1, :])                
    #             probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0)
                 
    #             topk_probs, topk_indices = torch.topk(probs, k)
                
    #             for i in range(k):
    #                 next_token = topk_indices[i].item()
    #                 next_score = score + topk_probs[i].item()
    #                 candidate = (seq + [next_token], next_score)
    #                 all_candidates.append(candidate)
                
    #         ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            
    #         beam = ordered[:k]
            
    #         all_ended = all(seq[-1] == self.endTokenIdx or seq[-1] == self.transTokenIdx
    #             for seq, _ in beam)
            
    #         if all_ended:
    #             break
            
    #     return beam[0][0]
