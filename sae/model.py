import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UNIT = "word" # unit for tokenization (char, word)
BATCH_SIZE = 128
EMBED_SIZE = 512
DROPOUT = 0.5
HIDDEN_SIZE = 500
BIDIRECTIONAL = True
NUM_DIRS = 2 if BIDIRECTIONAL else 1
VERBOSE = False
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
        self.rnn_type = "LSTM"
        self.num_layers = 1

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        self.pe = pos_encoder() # positional encoding
        self.layers = nn.ModuleList([enc_layer() for _ in range(self.num_layers)])
        self.rnn = rnn("LSTM", vocab_size, num_labels)
        self.fc = nn.Linear(HIDDEN_SIZE, num_labels)
        self.softmax = nn.LogSoftmax(1)

        if CUDA:
            lself = self.cuda()

    def forward(self, x):
        mask = maskset(x)
        # x = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        x = self.embed(x)
        h = x + self.pe(x.size(1))
        for layer in self.layers:
            h = layer(h, mask[0])
        h = self.rnn(h, mask)
        h = self.fc(h)
        y = self.softmax(h)
        return y

class enc_layer(nn.Module): # encoder layer
    def __init__(self):
        super().__init__()

        # architecture
        self.attn = attn_mh(EMBED_SIZE) # self-attention
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
    def __init__(self, dim):
        super().__init__()
        self.D = dim # dimension of model
        self.H = 8 # number of heads
        self.Dk = self.D // self.H # dimension of key
        self.Dv = self.D // self.H # dimension of value

        # architecture
        self.Wq = nn.Linear(self.D, self.H * self.Dk) # query
        self.Wk = nn.Linear(self.D, self.H * self.Dk) # key for attention distribution
        self.Wv = nn.Linear(self.D, self.H * self.Dv) # value for context representation
        self.Wo = nn.Linear(self.H * self.Dv, self.D)
        self.dropout = nn.Dropout(DROPOUT)
        self.norm = nn.LayerNorm(self.D)

    def attn_sdp(self, q, k, v, mask): # scaled dot-product attention
        c = np.sqrt(self.Dk) # scale factor
        a = torch.matmul(q, k.transpose(2, 3)) / c # compatibility function
        a = a.masked_fill(mask, -10000) # masking in log space
        a = F.softmax(a, -1)
        a = torch.matmul(a, v)
        return a # attention weights

    def forward(self, q, k, v, mask):
        x = q # identity
        q = self.Wq(q).view(BATCH_SIZE, -1, self.H, self.Dk).transpose(1, 2)
        k = self.Wk(k).view(BATCH_SIZE, -1, self.H, self.Dk).transpose(1, 2)
        v = self.Wv(v).view(BATCH_SIZE, -1, self.H, self.Dv).transpose(1, 2)
        z = self.attn_sdp(q, k, v, mask)
        z = z.transpose(1, 2).contiguous().view(BATCH_SIZE, -1, self.H * self.Dv)
        z = self.Wo(z)
        z = self.norm(x + self.dropout(z)) # residual connection and dropout
        return z

class ffn(nn.Module): # position-wise feed-forward networks
    def __init__(self, dim):
        super().__init__()

        # architecture
        self.layers = nn.Sequential(
            nn.Linear(EMBED_SIZE, dim),
            nn.ReLU(),
            nn.Linear(dim, EMBED_SIZE),
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.norm = nn.LayerNorm(EMBED_SIZE)

    def forward(self, x):
        z = self.layers(x)
        z = self.norm(x + self.dropout(z)) # residual connection and dropout
        return z

class rnn(nn.Module):
    def __init__(self, rnn_type, vocab_size, num_labels):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_layers = 2

        # architecture
        self.rnn = getattr(nn, rnn_type)(
            input_size = EMBED_SIZE,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = self.num_layers,
            batch_first = True,
            bidirectional = BIDIRECTIONAL
        )
        self.attn = attn_mh(HIDDEN_SIZE)

    def init_hidden(self): # initialize hidden states
        h = zeros(self.num_layers * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS) # hidden state
        if self.rnn_type == "LSTM":
            c = zeros(self.num_layers * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS) # cell state
            return (h, c)
        return h

    def forward(self, x, mask):
        self.hidden = self.init_hidden()
        x = nn.utils.rnn.pack_padded_sequence(x, mask[1], batch_first = True)
        ho, hc = self.rnn(x, self.hidden)
        ho, _ = nn.utils.rnn.pad_packed_sequence(ho, batch_first = True)
        hc = hc if self.rnn_type == "GRU" else hc[-1]
        hc = torch.cat([x for x in hc[-NUM_DIRS:]], 1)
        h = self.attn(hc, ho, ho, mask[0])
        return hc

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
    return (mask.view(BATCH_SIZE, 1, 1, -1), x.size(1) - mask.sum(1)) # set of mask and lengths
