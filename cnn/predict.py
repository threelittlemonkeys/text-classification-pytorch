import sys
import re
from model import *
from utils import *

def load_model():
    word_to_idx = load_word_to_idx(sys.argv[2])
    tag_to_idx = load_tag_to_idx(sys.argv[3])
    idx_to_tag = [tag for tag, _ in sorted(tag_to_idx.items(), key = lambda x: x[1])]
    model = cnn(len(word_to_idx), len(tag_to_idx))
    model.eval()
    if CUDA:
        model = model.cuda()
    print(model)
    load_checkpoint(sys.argv[1], model)
    return model, word_to_idx, tag_to_idx, idx_to_tag

def run_model(model, idx_to_tag, data):
    pred = []
    batch = []
    z = len(data)
    while len(data) < BATCH_SIZE:
        data.append(["", []])
    data.sort(key = lambda x: len(x[1]), reverse = True)
    batch_len = len(data[0][1])
    batch = [[SOS_IDX] + x + [PAD_IDX] * (batch_len - len(x)) + [EOS_IDX] for _, x in data]
    result = model(LongTensor(batch))
    for i in range(z):
        m = argmax(result[i])
        y = idx_to_tag[m]
        data[i] = (data[i][0], y)
    return data[:z]

def predict():
    data = []
    model, word_to_idx, tag_to_idx, idx_to_tag = load_model()
    fo = open(sys.argv[4])
    for line in fo:
        line = line.strip()
        x = tokenize(line, UNIT)
        x = [word_to_idx[i] if i in word_to_idx else UNK_IDX for i in x]
        data.append([line, x])
        if len(data) == BATCH_SIZE:
            result = run_model(model, idx_to_tag, data)
            for x in result:
                print(x)
            data = []
    fo.close()
    if len(data):
        result = run_model(model, idx_to_tag, data)
        for x in result:
            print(x)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model word_to_idx tag_to_idx test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    predict()
