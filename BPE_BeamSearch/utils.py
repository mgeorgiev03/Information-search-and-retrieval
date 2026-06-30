#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
##########################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import pickle
import re
from bpe import BPE 
from parameters import num_merges
import sys
import random
import nltk
from nltk.translate.bleu_score import corpus_bleu
from parameters import wordsFileName, bpeFileName
nltk.download('punkt')

class progressBar:
    def __init__(self ,barWidth = 50):
        self.barWidth = barWidth
        self.period = None
    def start(self, count):
        self.item=0
        self.period = int(count / self.barWidth)
        sys.stdout.write("["+(" " * self.barWidth)+"]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth+1))
    def tick(self):
        if self.item>0 and self.item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
        self.item += 1
    def stop(self):
        sys.stdout.write("]\n")

def readCorpus(fileName):
    ### Чете файл от изречения разделени с нов ред `\n`.
    ### fileName е името на файла, съдържащ корпуса
    ### връща списък от изречения, като всяко изречение е списък от думи
    print('Loading file:',fileName)
    return [ nltk.word_tokenize(line) for line in open(fileName) ]

def getDictionaryBPE(bpe, corpus, startToken, endToken, unkToken, padToken, transToken):

    bpe.getVocab(corpus, num_merges)
    bpeList = [startToken, endToken, unkToken, padToken, transToken] + bpe.symbols
    
    word2ind = { w:i for i,w in enumerate(bpeList)}
    
    return word2ind


def prepareDataBPE(bpe, sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, startToken, endToken, unkToken, padToken, transToken):

    sourceCorpus = readCorpus(sourceFileName)
    targetCorpus = readCorpus(targetFileName)

    #word2ind = getDictionaryBPE(bpe, sourceCorpus+targetCorpus, startToken, endToken, unkToken, padToken, transToken)
    word2ind = pickle.load(open(wordsFileName, 'rb')) 
    bpe.symbols = pickle.load(open(bpeFileName, 'rb'))
    bpe.symbols = [startToken, endToken, unkToken, padToken, transToken] + bpe.symbols

    trainCorpus = [ s + [transToken] + t for (s,t) in zip(sourceCorpus, targetCorpus)]
    trainCorpus = [ [startToken] + bpe.tokenizeBGEN(sent) + [endToken] for sent in trainCorpus]
    
    sourceDev = readCorpus(sourceDevFileName)
    targetDev = readCorpus(targetDevFileName)

    devCorpus = [ s + [transToken] + t for (s, t) in zip(sourceDev, targetDev)]
    devCorpus = [ [startToken] + bpe.tokenizeBGEN(sent) + [endToken] for sent in devCorpus]

    # trainCorpus = [ [ item for item in s if item != '</w>' ] for s in trainCorpus]
    # devCorpus   = [ [ item for item in s if item != '</w>' ] for s in devCorpus]

    print('Corpus loading completed.')
    return trainCorpus, devCorpus, word2ind

def postProcess(r, words):

    if r == []:
        return

    result = [words[i] for i in r[1:]]
    
    res = ""
    curr_word = ""

    if "</w>" in result[0]:
        res = result[0]
    else:
        curr_word = result[0]

    for w in result[1:]:
        if "</w>" in w:
            curr_word += w
            res += curr_word
            curr_word = ""
        else:
            curr_word += w
    
    res = res.replace("</w>", " ")
    res = re.sub(r' +(?=[.,\';])', '', res)
    #print(res)

    return res