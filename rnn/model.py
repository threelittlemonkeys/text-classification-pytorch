import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UNIT = "char" # unit for tokenization (char, word)
BATCH_SIZE = 128
EMBED_SIZE = 300
HIDDEN_SIZE = 1000
DROPOUT = 0.5
BIDIRECTIONAL = True
NUM_DIRS = 2 if BIDIRECTIONAL else 1
NUM_HEADS = 8
DK = HIDDEN_SIZE // NUM_HEADS # dimension of key
DV = HIDDEN_SIZE // NUM_HEADS # dimension of value
VERBOSE = False
SAVE_EVERY = 10

PAD = "<PAD>" # padding
UNK = "<UNK>" # unknown token

PAD_IDX = 0
UNK_IDX = 1

torch.manual_seed(1)
CUDA = torch.cuda.is_available()

class rnn(nn.Module):
    def __init__(self, rnn_type, vocab_size, num_labels):
        super().__init__()
        self.rnn_type = rnn_type

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        self.rnn1 = getattr(nn, rnn_type)(
            input_size = EMBED_SIZE,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            batch_first = True,
            bidirectional = BIDIRECTIONAL
        )
        self.rnn2 = getattr(nn, rnn_type)(
            input_size = HIDDEN_SIZE,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            batch_first = True,
            bidirectional = BIDIRECTIONAL
        )
        # self.attn = attn(HIDDEN_SIZE)
        # self.attn = attn(EMBED_SIZE + HIDDEN_SIZE * 2)
        self.attn = attn_mh()
        self.fc = nn.Linear(HIDDEN_SIZE, num_labels)
        self.softmax = nn.LogSoftmax(1)

        if CUDA:
            self = self.cuda()

    def init_hidden(self): # initialize hidden states
        h = zeros(NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS) # hidden state
        if self.rnn_type == "LSTM":
            c = zeros(NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS) # cell state
            return (h, c)
        return h

    def forward(self, x, mask):
        self.hidden1 = self.init_hidden()
        self.hidden2 = self.init_hidden()
        x = self.embed(x)
        h = nn.utils.rnn.pack_padded_sequence(x, mask[1], batch_first = True)
        h1, self.hidden1 = self.rnn1(h, self.hidden1)
        h2, self.hidden2 = self.rnn2(h1, self.hidden2)
        h = self.hidden2 if self.rnn_type == "GRU" else self.hidden2[-1]
        h = torch.cat([x for x in h[-NUM_DIRS:]], 1) # final cell state
        if self.attn:
            h1, _ = nn.utils.rnn.pad_packed_sequence(h1, batch_first = True)
            h2, _ = nn.utils.rnn.pad_packed_sequence(h2, batch_first = True)
            # global attention
            # h = self.attn(h, h2, mask[0])
            # h = self.attn(h, torch.cat((x, h1, h2), 2), mask[0])
            # multi-head attention
            h = self.attn(h, h2, h2, mask[0].view(BATCH_SIZE, 1, 1, -1))
        h = self.fc(h)
        y = self.softmax(h)
        return y

class attn(nn.Module): # global attention
    def __init__(self, attn_size):
        super().__init__()
        self.Va = None # attention weights

        # architecture
        self.Wa = nn.Linear(attn_size, 1)
        self.Wc = nn.Linear(HIDDEN_SIZE + attn_size, HIDDEN_SIZE)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, hc, ho, mask):
        a = self.Wa(ho).transpose(1, 2)
        a = a.masked_fill(mask.unsqueeze(1), -10000) # masking in log space
        a = self.Va = F.softmax(a, 2) # attention vector [B, 1, L]
        c = a.bmm(ho).squeeze(1) # context vector
        h = self.Wc(torch.cat((hc, self.dropout(c)), 1))
        return h

class attn_mh(nn.Module): # multi-head attention
    def __init__(self):
        super().__init__()
        self.Va = None # query-key attention weights

        # architecture
        self.Wq = nn.Linear(HIDDEN_SIZE, NUM_HEADS * DK) # query
        self.Wk = nn.Linear(HIDDEN_SIZE, NUM_HEADS * DK) # key for attention distribution
        self.Wv = nn.Linear(HIDDEN_SIZE, NUM_HEADS * DV) # value for context representation
        self.Wo = nn.Linear(NUM_HEADS * DV, HIDDEN_SIZE)
        self.dropout = nn.Dropout(DROPOUT)
        self.norm = nn.LayerNorm(HIDDEN_SIZE)

    def attn_sdp(self, q, k, v, mask): # scaled dot-product attention
        c = np.sqrt(DK) # scale factor
        a = torch.matmul(q, k.transpose(2, 3)) / c # compatibility function
        a = a.masked_fill(mask, -10000) # masking in log space
        a = self.Va = F.softmax(a, -1) # [B, NUM_HEADS, 1, L]
        a = torch.matmul(a, v) # [B, NUM_HEADS, 1, DV]
        return a # attention weights

    def forward(self, q, k, v, mask):
        x = q # identity
        q = self.Wq(q).view(BATCH_SIZE, -1, NUM_HEADS, DK).transpose(1, 2)
        k = self.Wk(k).view(BATCH_SIZE, -1, NUM_HEADS, DK).transpose(1, 2)
        v = self.Wv(v).view(BATCH_SIZE, -1, NUM_HEADS, DV).transpose(1, 2)
        z = self.attn_sdp(q, k, v, mask)
        z = z.transpose(1, 2).contiguous().view(BATCH_SIZE, -1, NUM_HEADS * DV)
        z = self.Wo(z).squeeze(1)
        z = self.norm(x + self.dropout(z)) # residual connection and dropout
        return z

def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if CUDA else x

def argmax(x):
    return torch.max(x, 0)[1].tolist() # for 1D tensor

def maskset(x):
    mask = x.data.eq(PAD_IDX)
    return (mask, x.size(1) - mask.sum(1)) # set of mask and lengths
