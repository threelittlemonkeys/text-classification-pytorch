import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 128
EMBED_SIZE = 100
HIDDEN_SIZE = 1000
NUM_LAYERS = 2
DROPOUT = 0.5
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

class rnn(nn.Module):
    def __init__(self, rnn_type, vocab_size, num_labels):
        super().__init__()
        self.rnn_type = rnn_type
        fc_hidden_size = (EMBED_SIZE + HIDDEN_SIZE * 2) if RESIDUAL else HIDDEN_SIZE

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        self.rnn = getattr(nn, rnn_type)( # LSTM or GRU
            input_size = EMBED_SIZE,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            batch_first = True,
            bidirectional = BIDIRECTIONAL
        )
        self.attn = None # attn(EMBED_SIZE + HIDDEN_SIZE * 2)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(fc_hidden_size, num_labels)
        self.softmax = nn.LogSoftmax(1)

        if CUDA:
            self = self.cuda()

    def init_hidden(self, rnn_type): # initialize hidden states
        h = zeros(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS) # hidden state
        if rnn_type == "LSTM":
            c = zeros(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS) # cell state
            return (h, c)
        return h

    def forward(self, x, mask):
        self.hidden = self.init_hidden(self.rnn_type)
        x = self.embed(x)
        h = nn.utils.rnn.pack_padded_sequence(x, mask[1], batch_first = True)
        h, self.hidden = self.rnn(h, self.hidden)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)
        if self.rnn_type == "LSTM":
            self.hidden = self.hidden[-1] # cell state
        h1 = torch.cat((self.hidden[0], self.hidden[1]), 1)
        h2 = torch.cat((self.hidden[2], self.hidden[3]), 1)
        print(x.size())
        print(h1.size())
        exit()
        if self.attn:
            h = self.attn(torch.cat((x, h1, h2), 2), mask[0])
        else:
            h = h
        h = self.dropout(h)
        h = self.fc(h).squeeze(1)
        h = self.softmax(h)
        return h

class attn(nn.Module): # attention layer
    def __init__(self, hidden_size):
        super().__init__()

        # architecture
        self.Wa = nn.Linear(hidden_size, 1)

    def align(self, h, mask):
        a = self.Wa(h) # [B, L, 1]
        a = a.masked_fill(mask.unsqueeze(2), -10000) # masking in log space
        a = F.softmax(a, 1)
        return a # alignment weights

    def forward(self, h, mask):
        a = self.align(h, mask) # alignment vector
        v = (a * h).sum(1) # representation vector
        return v

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

def gather_output(x, mask): # gather the last RNN output
    idx = mask.view(-1, 1, 1).expand(-1, -1, x.size(2)) - 1 # [B, 1, H]
    return x.gather(1, idx)

def maskset(x):
    mask = x.data.eq(PAD_IDX)
    return (mask, x.size(1) - mask.sum(1)) # set of mask and lengths
