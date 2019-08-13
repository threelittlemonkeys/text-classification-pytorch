from utils import *
from embedding import embed

class cnn(nn.Module):
    def __init__(self, char_vocab_size, word_vocab_size, num_labels):
        super().__init__()

        # architecture
        self.embed = embed(char_vocab_size, word_vocab_size)
        self.conv = nn.ModuleList([nn.Conv2d(
            in_channels = 1, # Ci
            out_channels = NUM_FEATMAPS, # Co
            kernel_size = (i, sum(EMBED.values())) # height, width
        ) for i in KERNEL_SIZES]) # num_kernels (K)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(len(KERNEL_SIZES) * NUM_FEATMAPS, num_labels)
        self.softmax = nn.LogSoftmax(1)

        if CUDA:
            self = self.cuda()

    def forward(self, xc, xw):
        x = self.embed(xc, xw) # [batch_size (B), seq_len (L), embed_size (H)]
        x = x.unsqueeze(1) # [B, Ci, L, H]
        h = [conv(x) for conv in self.conv] # [B, Co, L, 1] * K
        h = [F.relu(k).squeeze(3) for k in h] # [B, Co, L] * K
        h = [F.max_pool1d(k, k.size(2)).squeeze(2) for k in h] # [B, Co] * K
        h = torch.cat(h, 1) # [B, Co * K]
        h = self.dropout(h)
        h = self.fc(h) # fully connected layer
        y = self.softmax(h)
        return y
