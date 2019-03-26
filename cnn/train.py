import sys
import re
import time
from model import *
from utils import *
from os.path import isfile

def load_data():
    data = []
    bxc = [] # character sequence batch
    bxw = [] # word sequence batch
    by = [] # label batch
    cti = load_tkn_to_idx(sys.argv[2]) # char_to_idx
    itw = load_idx_to_tkn(sys.argv[3]) # idx_to_word
    tti = load_tkn_to_idx(sys.argv[4]) # tag_to_idx
    print("loading %s" % sys.argv[5])
    fo = open(sys.argv[5], "r")
    for line in fo:
        line = line.strip()
        *x, y = [x.split(":") for x in line.split(" ")]
        xc, xw = zip(*[(list(map(int, xc.split("+"))), int(xw)) for xc, xw in x])
        bxc.append(xc)
        bxw.append(xw)
        by.append(int(y[0]))
        if len(by) == BATCH_SIZE:
            bxc, bxw = batchify(bxc, bxw, max(KERNEL_SIZES))
            data.append((LongTensor(bxc), LongTensor(bxw), LongTensor(by)))
            bxc = []
            bxw = []
            by = []
    fo.close()
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, len(cti), len(itw), len(tti)

def train():
    num_epochs = int(sys.argv[6])
    data, char_vocab_size, word_vocab_size, num_labels = load_data()
    model = cnn(char_vocab_size, word_vocab_size, num_labels)
    print(model)
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    epoch = load_checkpoint(sys.argv[1], model) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        loss_sum = 0
        timer = time.time()
        for xc, xw, y in data:
            model.zero_grad()
            loss = F.nll_loss(model(xc, xw), y) # forward pass and compute loss
            loss.backward() # compute gradients
            optim.step() # update parameters
            loss = loss.tolist()
            loss_sum += loss
        timer = time.time() - timer
        loss_sum /= len(data)
        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, model, ei, loss_sum, timer)

if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx training_data num_epoch" % sys.argv[0])
    print("cuda: %s" % CUDA)
    train()
