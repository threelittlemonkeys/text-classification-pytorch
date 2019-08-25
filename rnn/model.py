from utils import *
from embedding import embed

class rnn(nn.Module):
    def __init__(self, char_vocab_size, word_vocab_size, num_labels):
        super().__init__()

        # architecture
        self.embed = embed(char_vocab_size, word_vocab_size)
        self.rnn1 = getattr(nn, RNN_TYPE)(
            input_size = sum(EMBED.values()),
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = 1,
            batch_first = True,
            bidirectional = NUM_DIRS == 2
        )
        self.rnn2 = getattr(nn, RNN_TYPE)(
            input_size = HIDDEN_SIZE,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = 1,
            batch_first = True,
            bidirectional = NUM_DIRS == 2
        )
        self.attn = None
        if ATTN == "attn": # global attention
            self.attn = attn(HIDDEN_SIZE)
        if ATTN == "attn-rc": # global attention with residual connection
            self.attn = attn(sum(EMBED.values()) + HIDDEN_SIZE * 2)
        if ATTN == "mh-attn": # multi-head attention
            self.attn = attn_mh()
        self.fc = nn.Linear(HIDDEN_SIZE, num_labels)
        self.softmax = nn.LogSoftmax(1)

        if CUDA:
            self = self.cuda()

    def init_state(self): # initialize RNN states
        args = (1 * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS) # NUM_LAYERS = 1
        hs = zeros(*args) # hidden state
        if RNN_TYPE == "LSTM":
            cs = zeros(*args) # LSTM cell state
            return (hs, cs)
        return hs

    def forward(self, xc, xw, mask):
        s1 = self.init_state()
        s2 = self.init_state()
        x = self.embed(xc, xw)
        x = nn.utils.rnn.pack_padded_sequence(x, mask[1], batch_first = True)
        h1, s1 = self.rnn1(x, s1)
        h2, s2 = self.rnn2(h1, s2)
        h = s2 if RNN_TYPE == "GRU" else s2[-1]
        h = torch.cat([x for x in h[-NUM_DIRS:]], 1) # final cell state
        if self.attn:
            h1, _ = nn.utils.rnn.pad_packed_sequence(h1, batch_first = True)
            h2, _ = nn.utils.rnn.pad_packed_sequence(h2, batch_first = True)
            if ATTN == "attn":
                h = self.attn(h, h2, mask[0])
            if ATTN == "attn-rc":
                h = self.attn(h, torch.cat((x, h1, h2), 2), mask[0])
            if ATTN == "mh-attn":
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
        c = a.bmm(ho).squeeze(1) # context vector [B, H]
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
        a = F.softmax(a, 3) # [B, NUM_HEADS, 1, L]
        self.Va = a.squeeze(2)
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
