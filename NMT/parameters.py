import torch

sourceFileName = 'en_bg_data/train.bg'
targetFileName = 'en_bg_data/train.en'
sourceDevFileName = 'en_bg_data/dev.bg'
targetDevFileName = 'en_bg_data/dev.en'

corpusFileName = 'corpusData'
wordsFileName = 'wordsData'
modelFileName = 'NMTmodel'

device = torch.device("cuda:0")
#device = torch.device("cpu")

n_head = 4
d_model = 256
d_ff = d_model * 4
transformer_layers = 6

learning_rate = 0.01
batchSize = 50
clip_grad = 5.0

maxEpochs = 10
log_every = 10
test_every = 2000
