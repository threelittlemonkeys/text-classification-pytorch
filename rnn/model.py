import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.attn = attn(HIDDEN_SIZE)
        # self.attn = attn(EMBED_SIZE + HIDDEN_SIZE * 2)
        # self.attn_mh = attn_mh()
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE, num_labels)
        self.softmax = nn.LogSoftmax(1)

        if CUDA:
            self = self.cuda()

    def init_hidden(self, rnn_type): # initialize hidden states
        h = zeros(NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS) # hidden state
        if rnn_type == "LSTM":
            c = zeros(NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS) # cell state
            return (h, c)
        return h

    def forward(self, x, mask):
        self.hidden1 = self.init_hidden(self.rnn_type)
        self.hidden2 = self.init_hidden(self.rnn_type)
        x = self.embed(x)
        h = nn.utils.rnn.pack_padded_sequence(x, mask[1], batch_first = True)
        h1, self.hidden1 = self.rnn1(h, self.hidden1)
        h2, self.hidden2 = self.rnn2(h1, self.hidden2)
        h = self.hidden2 if self.rnn_type == "GRU" else self.hidden2[-1]
        h = torch.cat([x for x in h[-NUM_DIRS:]], 1) # final cell state
        if self.attn:
            h1, _ = nn.utils.rnn.pad_packed_sequence(h1, batch_first = True)
            h2, _ = nn.utils.rnn.pad_packed_sequence(h2, batch_first = True)
            h = self.attn(h, h2, mask[0])
            # h = self.attn(h, torch.cat((x, h1, h2), 2), mask[0])
        h = self.dropout(h)
        h = self.fc(h)
        y = self.softmax(h)
        return y

class attn(nn.Module): # attention
    def __init__(self, attn_size):
        super().__init__()

        # architecture
        self.Wa = nn.Linear(attn_size, 1)
        self.Wc = nn.Linear(HIDDEN_SIZE + attn_size, HIDDEN_SIZE)

    def align(self, h, mask):
        a = self.Wa(h).transpose(1, 2) # [B, 1, L]
        a = a.masked_fill(mask.unsqueeze(1), -10000) # masking in log space
        a = F.softmax(a, 2)
        return a # alignment weights

    def forward(self, h, ht, mask):
        a = self.align(ht, mask) # alignment vector
        v = a.bmm(ht).squeeze(1) # representation vector
        h = self.Wc(torch.cat((h, v), 1)) # attentional vector
        return h

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

def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if CUDA else x

def scalar(x):
    return x.view(-1).data.tolist()[0]

def argmax(x):
    return scalar(torch.max(x, 0)[1]) # for 1D tensor

def maskset(x):
    mask = x.data.eq(PAD_IDX)
    return (mask, x.size(1) - mask.sum(1)) # set of mask and lengths
