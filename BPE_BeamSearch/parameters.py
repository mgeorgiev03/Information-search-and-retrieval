import torch

sourceFileName = 'en_bg_data/train.bg'
targetFileName = 'en_bg_data/train.en'
sourceDevFileName = 'en_bg_data/dev.bg'
targetDevFileName = 'en_bg_data/dev.en'

corpusFileName = 'corpusData'
wordsFileName = 'wordsData'
modelFileName = 'NMTmodel'

bpeFileName = 'bpeData'

#device = torch.device("cuda:0")
device = torch.device("cpu")

n_head = 4 # num of attn heads
num_merges = 8000

learning_rate = 0.01
batchSize = 64
d_model = 256
d_ff = d_model*4
transformer_layers = 6
clip_grad = 5.0

maxEpochs = 10
log_every = 10
test_every = 2000
