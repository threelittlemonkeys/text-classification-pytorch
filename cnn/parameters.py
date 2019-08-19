import math
import torch
import torch.nn as nn
import torch.nn.functional as F

UNIT = "word" # unit of tokenization (char, word)
BATCH_SIZE = 64
EMBED = {"char-cnn": 50, "lookup": 250} # embeddings (char-cnn, char-rnn, lookup, sae)
NUM_FEATMAPS = 100 # feature maps generated by each kernel
KERNEL_SIZES = [2, 3, 4]
DROPOUT = 0.5
LEARNING_RATE = 1e-4
VERBOSE = False
EVAL_EVERY = 10
SAVE_EVERY = 10

PAD = "<PAD>" # padding
SOS = "<SOS>" # start of sequence
EOS = "<EOS>" # end of sequence
UNK = "<UNK>" # unknown token

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

CUDA = torch.cuda.is_available()
torch.manual_seed(0) # for reproducibility

DELIM = "\t" # delimiter
KEEP_IDX = False # use the existing indices when preparing additional data
NUM_DIGITS = 4 # number of digits to print
