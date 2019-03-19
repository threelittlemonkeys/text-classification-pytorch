import sys
import re
import time
from model import *
from utils import *
from evaluate import *
from os.path import isfile

def load_data():
    data = []
    batch_x = []
    batch_y = []
    batch_len = 0 # maximum sequence length of a mini-batch
    print("loading data...")
    word_to_idx = load_tkn_to_idx(sys.argv[2])
    tag_to_idx = load_tkn_to_idx(sys.argv[3])
    idx_to_word = idx_to_tkn(word_to_idx)
    idx_to_tag = idx_to_tkn(tag_to_idx)
    fo = open(sys.argv[4], "r")
    for line in fo:
        line = line.strip()
        seq = [int(i) for i in line.split(" ")]
        label = seq.pop()
        if len(batch_x) == 0: # the first line has the maximum sequence length
            batch_len = len(seq)
        pad = [PAD_IDX] * (batch_len - len(seq))
        batch_x.append(seq + pad)
        batch_y.append(label)
        if len(batch_x) == BATCH_SIZE:
            data.append((LongTensor(batch_x), LongTensor(batch_y))) # append a mini-batch
            batch_x = []
            batch_y = []
    fo.close()
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, word_to_idx, tag_to_idx, idx_to_word, idx_to_tag

def train():
    num_epochs = int(sys.argv[-1])
    data, word_to_idx, tag_to_idx, idx_to_word, idx_to_tag = load_data()
    model = rnn(len(word_to_idx), len(tag_to_idx))
    print(model)
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    epoch = load_checkpoint(sys.argv[1], model) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        ii = 0
        loss_sum = 0
        timer = time.time()
        for x, y in data:
            ii += 1
            model.zero_grad()
            mask = maskset(x)
            loss = F.nll_loss(model(x, mask), y)
            loss.backward()
            optim.step()
            loss = loss.tolist()
            loss_sum += loss
            # print("epoch = %d, iteration = %d, loss = %f" % (ei, ii, loss))
        timer = time.time() - timer
        loss_sum /= len(data)
        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, model, ei, loss_sum, timer)
        if EVAL_EVERY and (ei % EVAL_EVERY == 0 or ei == epoch + num_epochs):
            args = [model, word_to_idx, tag_to_idx, idx_to_word, idx_to_tag]
            evaluate(predict(sys.argv[5], True, *args), True)
            print()
if __name__ == "__main__":
    if len(sys.argv) not in [6, 7]:
        sys.exit("Usage: %s model word_to_idx tag_to_idx training_data (validation_data) num_epoch" % sys.argv[0])
    print("cuda: %s" % CUDA)
    train()
