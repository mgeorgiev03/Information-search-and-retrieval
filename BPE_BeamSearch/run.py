#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import sys
import numpy as np
import torch
import math 
import pickle
import time

from nltk.translate.bleu_score import corpus_bleu

import utils
import model
from parameters import *
from bpe import BPE

startToken = '<S>'
startTokenIdx = 0

endToken = '</S>'
endTokenIdx = 1

unkToken = '<UNK>'
unkTokenIdx = 2

padToken = '<PAD>'
padTokenIdx = 3

transToken = '<TRANS>'
transTokenIdx = 4

def perplexity(nmt, test, batchSize):
    testSize = len(test)
    H = 0.
    c = 0
    for b in range(0,testSize,batchSize):
        batch = test[b:min(b+batchSize, testSize)]
        l = sum(len(s)-1 for s in batch)
        c += l
        with torch.no_grad():
            H += l * nmt(batch)
    return math.exp(H/c)
   
def beam(probs, k):
        seqs = [[list(), 0.0]]
        for row in probs:
            all_candidates = list()
            for i in range(len(seqs)):
                seq, score = seqs[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score - torch.log(row[j])]
                    all_candidates.append(candidate)

            ordered = sorted(all_candidates, key=lambda x: x[1])
            seqs = ordered[:k]
        return seqs

if len(sys.argv)>1 and sys.argv[1] == 'check':

    # bpe = BPE()
    # bpe.symbols = pickle.load(open(bpeFileName, 'rb'))
    # bpe.symbols = [startToken, endToken, unkToken, padToken, transToken] + bpe.symbols

    # print(bpe.symbols[:300])

    # (trainCorpus,devCorpus) = pickle.load(open(corpusFileName, 'rb'))
    # word2ind = pickle.load(open(wordsFileName, 'rb')) 

    # words = list(word2ind)
    # e = [ words[i] for i in trainCorpus[1] ]
    # print(e)    

    
    probs = pickle.load(open("probs.txt", 'rb'))
    print(np.shape(probs))
    next_token = torch.argmax(probs, dim=1).item()
    # remove argmax 
    probs[0, next_token] = 0
    # mask = torch.ones(np.shape(probs), dtype=torch.bool)  
    # mask[0, next_token] = False
    # probs = probs[mask]
    print(np.shape(probs))

    # draw argmax again
    # do it k times
    # make k sequences
    # beam search them
 

    
if len(sys.argv)>1 and sys.argv[1] == 'prepare':
    bpe = BPE()
    trainCorpus, devCorpus, word2ind = utils.prepareDataBPE(bpe, sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, startToken, endToken, unkToken, padToken, transToken)
    
    trainCorpus = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in trainCorpus ]
    devCorpus = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in devCorpus ]

    pickle.dump((trainCorpus, devCorpus), open(corpusFileName, 'wb'))
    # pickle.dump(word2ind, open(wordsFileName, 'wb'))
    # pickle.dump(bpe.symbols, open(bpeFileName, 'wb'))
    print('Data prepared.')

if len(sys.argv)>1 and (sys.argv[1] == 'train' or sys.argv[1] == 'extratrain'):

    bpe = BPE()
    bpe.symbols = pickle.load(open(bpeFileName, 'rb'))
    bpe.symbols = [startToken, endToken, unkToken, padToken, transToken] + bpe.symbols

    (trainCorpus,devCorpus) = pickle.load(open(corpusFileName, 'rb'))
    word2ind = pickle.load(open(wordsFileName, 'rb'))

    nmt = model.LanguageModel(n_head, d_model, word2ind, unkToken, padToken, transToken, endToken).to(device)
    optimizer = torch.optim.Adam(nmt.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate, 
        steps_per_epoch=len(trainCorpus) // batchSize, 
        epochs=maxEpochs,
        pct_start=0.1 
    )

    if sys.argv[1] == 'extratrain':
        nmt.load(modelFileName)
        (iter,bestPerplexity,learning_rate,osd) = torch.load(modelFileName + '.optim')
        optimizer.load_state_dict(osd)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    else:
        bestPerplexity = math.inf
        iter = 0

    idx = np.arange(len(trainCorpus), dtype='int32')
    nmt.train()
    beginTime = time.time()
    for epoch in range(maxEpochs):
        np.random.shuffle(idx)
        words = 0
        trainTime = time.time()
        for b in range(0, len(idx), batchSize):
			#############################################################################
			### Може да се наложи да се променя скоростта на спускане learning_rate в зависимост от итерацията
			#############################################################################
            iter += 1
            batch = [ trainCorpus[i] for i in idx[b:min(b+batchSize, len(idx))] ]
            
            words += sum( len(s)-1 for s in batch )
            H = nmt(batch)
            optimizer.zero_grad()
            H.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(nmt.parameters(), clip_grad)
            optimizer.step()
            if iter % log_every == 0:
                print("Iteration:",iter,"Epoch:",epoch+1,'/',maxEpochs,", Batch:",b//batchSize+1, '/', len(idx) // batchSize+1, ", loss: ",H.item(), "words/sec:",words / (time.time() - trainTime), "time elapsed:", (time.time() - beginTime) )
                trainTime = time.time()
                words = 0
                
            if iter % test_every == 0:
                nmt.eval()
                currentPerplexity = perplexity(nmt, devCorpus, batchSize)
                nmt.train()
                print('Current model perplexity: ',currentPerplexity)

                if currentPerplexity < bestPerplexity:
                    bestPerplexity = currentPerplexity
                    print('Saving new best model.')
                    nmt.save(modelFileName)
                    torch.save((iter,bestPerplexity,learning_rate,optimizer.state_dict()), modelFileName + '.optim')

    print('reached maximum number of epochs!')
    nmt.eval()
    currentPerplexity = perplexity(nmt, devCorpus, batchSize)
    print('Last model perplexity: ',currentPerplexity)
        
    if currentPerplexity < bestPerplexity:
        bestPerplexity = currentPerplexity
        print('Saving last model.')
        nmt.save(modelFileName)
        torch.save((iter,bestPerplexity,learning_rate,optimizer.state_dict()), modelFileName + '.optim')

if len(sys.argv)>3 and sys.argv[1] == 'perplexity':
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    
    bpe = BPE()
    bpe.symbols = pickle.load(open(bpeFileName, 'rb'))
    bpe.symbols = [startToken, endToken, unkToken, padToken, transToken] + bpe.symbols

    nmt = model.LanguageModel(n_head, d_model, word2ind, unkToken, padToken, transToken, endToken).to(device)
    nmt.load(modelFileName)
    
    sourceTest = utils.readCorpus(sys.argv[2])
    targetTest = utils.readCorpus(sys.argv[3])

    test = [ s + [transToken] + t for (s,t) in zip(sourceTest, targetTest)]
    test = [ bpe.tokenizeBGEN(sent) for sent in test]
    test = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in test ]

    nmt.eval()
    print('Model perplexity: ', perplexity(nmt, test, batchSize))

if len(sys.argv)>3 and sys.argv[1] == 'translate':
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    words = list(word2ind)

    bpe = BPE()
    bpe.symbols = pickle.load(open(bpeFileName, 'rb'))
    bpe.symbols = [startToken, endToken, unkToken, padToken, transToken] + bpe.symbols

    sourceTest = utils.readCorpus(sys.argv[2])
    test = [ bpe.tokenizeBG(s) for s in sourceTest]
    test = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in test ]
    
    nmt = model.LanguageModel(n_head, d_model, word2ind, unkToken, padToken, transToken, endToken).to(device)
    nmt.load(modelFileName)

    nmt.eval()
    file = open(sys.argv[3],'w')
    pb = utils.progressBar()
    pb.start(len(test))
    for s in test:
        r=nmt.generate(s)
        result = utils.postProcess(r, words)
        # st = r.index(transTokenIdx)
        # result = [words[i] for i in r[st+1:-1]]
        # result = [ i.replace("</w>", "") for i in result ]
        file.write(' '.join(result)+"\n")
        pb.tick()
    pb.stop()

if len(sys.argv)>2 and sys.argv[1] == 'generate':
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    words = list(word2ind)
    
    bpe = BPE()
    bpe.symbols = pickle.load(open(bpeFileName, 'rb'))
    bpe.symbols = [startToken, endToken, unkToken, padToken, transToken] + bpe.symbols

    test = sys.argv[2].split()
    test = bpe.tokenizeBG(test)
    test = [word2ind.get(w,unkTokenIdx) for w in test]

    nmt = model.LanguageModel(n_head, d_model, word2ind, unkToken, padToken, transToken, endToken).to(device)
    nmt.load(modelFileName)

    nmt.eval()
    r=nmt.generate(test)
    result = utils.postProcess(r, words) 
    print(result)
    # for i in range(len(r)):
    #     result = utils.postProcess(r[i][0], words)   
    #     print(result)
    #     print(r[i][1])

if len(sys.argv)>3 and sys.argv[1] == 'bleu':
    ref = [[s] for s in utils.readCorpus(sys.argv[2])]
    hyp = utils.readCorpus(sys.argv[3])

    bleu_score = corpus_bleu(ref, hyp)
    print('Corpus BLEU: ', (bleu_score * 100))
