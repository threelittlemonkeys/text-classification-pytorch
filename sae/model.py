import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UNIT = "char" # unit for tokenization (char, word)
BATCH_SIZE = 128
EMBED_SIZE = 64
NUM_LAYERS = 4
NUM_HEADS = 8 # number of heads
DK = EMBED_SIZE // NUM_HEADS # dimension of key
DV = EMBED_SIZE // NUM_HEADS # dimension of value
DROPOUT = 0.5
VERBOSE = True
SAVE_EVERY = 10

PAD = "<PAD>" # padding
UNK = "<UNK>" # unknown token

PAD_IDX = 0
UNK_IDX = 1

torch.manual_seed(1)
CUDA = torch.cuda.is_available()

class sae(nn.Module): # self attentive encoder
    def __init__(self, vocab_size, num_labels):
        super().__init__()

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        self.pe = pos_encoder() # positional encoding
        self.layers = nn.ModuleList([enc_layer() for _ in range(NUM_LAYERS)])
        self.fc = nn.Linear(EMBED_SIZE, num_labels)
        self.softmax = nn.LogSoftmax(1)

        if CUDA:
            lself = self.cuda()

    def forward(self, x, mask):
        x = self.embed(x)
        h = x + self.pe(x.size(1))
        for layer in self.layers:
            h = layer(h, mask)
        h = torch.sum(h, 1)
        h = self.fc(h)
        y = self.softmax(h)
        return y

class enc_layer(nn.Module): # encoder layer
    def __init__(self):
        super().__init__()

        # architecture
        self.attn = attn_mh() # self-attention
        self.ffn = ffn(2048)

    def forward(self, x, mask):
        z = self.attn(x, x, x, mask)
        z = self.ffn(z)
        return z

class pos_encoder(nn.Module): # positional encoding
    def __init__(self, maxlen = 1000):
        super().__init__()
        self.pe = Tensor(maxlen, EMBED_SIZE)
        pos = torch.arange(0, maxlen, 1.).unsqueeze(1)
        k = torch.exp(np.log(10000) * -torch.arange(0, EMBED_SIZE, 2.) / EMBED_SIZE)
        self.pe[:, 0::2] = torch.sin(pos * k)
        self.pe[:, 1::2] = torch.cos(pos * k)

    def forward(self, n):
        return self.pe[:n]

class attn_mh(nn.Module): # multi-head attention
    def __init__(self):
        super().__init__()

        # architecture
        self.Wq = nn.Linear(EMBED_SIZE, NUM_HEADS * DK) # query
        self.Wk = nn.Linear(EMBED_SIZE, NUM_HEADS * DK) # key for attention distribution
        self.Wv = nn.Linear(EMBED_SIZE, NUM_HEADS * DV) # value for context representation
        self.Wo = nn.Linear(NUM_HEADS * DV, EMBED_SIZE)
        self.dropout = nn.Dropout(DROPOUT)
        self.norm = nn.LayerNorm(EMBED_SIZE)

    def attn_sdp(self, q, k, v, mask): # scaled dot-product attention
        c = np.sqrt(DK) # scale factor
        a = torch.matmul(q, k.transpose(2, 3)) / c # compatibility function
        a = a.masked_fill(mask, -10000) # masking in log space
        a = F.softmax(a, -1)
        a = torch.matmul(a, v)
        return a # attention weights

    def forward(self, q, k, v, mask):
        x = q # identity
        q = self.Wq(q).view(BATCH_SIZE, -1, NUM_HEADS, DK).transpose(1, 2)
        k = self.Wk(k).view(BATCH_SIZE, -1, NUM_HEADS, DK).transpose(1, 2)
        v = self.Wv(v).view(BATCH_SIZE, -1, NUM_HEADS, DV).transpose(1, 2)
        z = self.attn_sdp(q, k, v, mask)
        z = z.transpose(1, 2).contiguous().view(BATCH_SIZE, -1, NUM_HEADS * DV)
        z = self.Wo(z)
        z = self.norm(x + self.dropout(z)) # residual connection and dropout
        return z

class ffn(nn.Module): # position-wise feed-forward networks
    def __init__(self, d):
        super().__init__()

        # architecture
        self.layers = nn.Sequential(
            nn.Linear(EMBED_SIZE, d),
            nn.ReLU(),
            nn.Linear(d, EMBED_SIZE),
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.norm = nn.LayerNorm(EMBED_SIZE)

    def forward(self, x):
        z = self.layers(x)
        z = self.norm(x + self.dropout(z)) # residual connection and dropout
        return z

def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def argmax(x):
    return torch.max(x, 0)[1].tolist() # for 1D tensor
