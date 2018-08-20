import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ_LEN = 1024 # maximum length of an input sequence
BATCH_SIZE = 64
EMBED_SIZE = 16
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1 # KERNEL_SIZE - 2
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
VERBOSE = False
SAVE_EVERY = 10

PAD = "<PAD>" # padding
PAD_IDX = 0

torch.manual_seed(1)
CUDA = torch.cuda.is_available()

class vdcnn(nn.Module):
    def __init__(self, vocab_size, num_labels):
        super().__init__()
        self.k = 8 # k-max pooling

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        self.conv = nn.Conv1d(EMBED_SIZE, 64, KERNEL_SIZE, STRIDE, PADDING)
        self.res_blocks = nn.Sequential( # residual blocks
            res_block(64, 64),
            res_block(64, 64, "vgg"),
            res_block(64, 128),
            res_block(128, 128, "vgg"),
            res_block(128, 256),
            res_block(256, 256, "vgg"),
            res_block(256, 512),
            res_block(512, 512)
        )
        self.fc = nn.Sequential( # fully connected layers
            nn.Linear(512 * self.k, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_labels)
        )
        self.softmax = nn.LogSoftmax(1)

        if CUDA:
            self = self.cuda()

    def forward(self, x):
        x = self.embed(x) # embedding
        x = x.transpose(1, 2) # [batch_size (N), num_feature_maps (D), seq_len (L)]
        h = self.conv(x) # temporal convolution
        h = self.res_blocks(h) # residual blocks
        h = h.topk(self.k)[0].view(BATCH_SIZE, -1) # k-max pooling
        h = self.fc(h) # fully connected layers
        y = self.softmax(h)
        return y

class res_block(nn.Module): # residual block
    def __init__(self, in_channels, out_channels, downsample = None):
        super().__init__()
        first_stride = 2 if downsample == "resnet" else 1
        pool_stride = 2 if downsample else 1

        # architecture
        self.conv_block = conv_block(in_channels, out_channels, first_stride)
        self.pool = None
        if downsample == "kmax": # k-max pooling (Kalchbrenner et al 2014)
            self.pool = lambda x: x.topk(x.size(2) // 2)[0]
        elif downsample == "vgg": # VGG-like
            self.pool = nn.MaxPool1d(KERNEL_SIZE, pool_stride, PADDING)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1, pool_stride)

    def forward(self, x):
        y = self.conv_block(x)
        if self.pool:
            y = self.pool(y)
        y += self.shortcut(x) # ResNet shortcut connections
        return y

class conv_block(nn.Module): # convolutional block
    def __init__(self, in_channels, out_channels, first_stride):
        super().__init__()

        # architecture
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, KERNEL_SIZE, first_stride, PADDING),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.sequential(x)

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def scalar(x):
    return x.view(-1).data.tolist()[0]

def argmax(x):
    return scalar(torch.max(x, 0)[1]) # for 1D tensor
