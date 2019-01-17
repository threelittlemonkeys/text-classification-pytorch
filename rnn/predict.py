import sys
import re
from model import *
from utils import *

def load_model():
    word_to_idx = load_word_to_idx(sys.argv[2])
    tag_to_idx = load_tag_to_idx(sys.argv[3])
    idx_to_word = [x for x, _ in sorted(word_to_idx.items(), key = lambda x: x[1])]
    idx_to_tag = [x for x, _ in sorted(tag_to_idx.items(), key = lambda x: x[1])]
    model = rnn("LSTM", len(word_to_idx), len(tag_to_idx))
    print(model)
    model.eval()
    load_checkpoint(sys.argv[1], model)
    return model, word_to_idx, tag_to_idx, idx_to_word, idx_to_tag

def run_model(model, idx_to_word, idx_to_tag, data):
    z = len(data)
    while len(data) < BATCH_SIZE:
        data.append([-1, "", [UNK_IDX], ""])
    data.sort(key = lambda x: -len(x[2]))
    batch_len = len(data[0][2])
    batch = LongTensor([x[2] + [PAD_IDX] * (batch_len - len(x[2])) for x in data])
    mask = maskset(batch)
    result = model(batch, mask)
    if VERBOSE:
        Va = model.attn.Va.squeeze(2).tolist() # attention weights
    for i in range(z):
        y = idx_to_tag[argmax(result[i])]
        data[i].append(y)
        if VERBOSE:
            print(data[i][1])
            y = enumerate(result[i].exp().tolist())
            for a, b in sorted(y, key = lambda x: -x[1]):
                print(idx_to_tag[a], round(b, 4))
            print(mat2csv(heatmap(Va[i], data[i][2], idx_to_word))) # attention heatmap
    for x in [x[1:] for x in sorted(data[:z])]:
        print(x)
    return [(x[1], *x[3:]) for x in sorted(data[:z])]

def predict(lb = False):
    idx = 0
    data = []
    result = []
    model, word_to_idx, tag_to_idx, idx_to_word, idx_to_tag = load_model()
    fo = open(sys.argv[4])
    for line in fo:
        line = line.strip()
        line, y = line.split("\t") if lb else [line, ""]
        x = tokenize(line, UNIT)
        x = [word_to_idx[i] if i in word_to_idx else UNK_IDX for i in x]
        data.append([idx, line, x, y])
        if len(data) == BATCH_SIZE:
            result.extend(run_model(model, idx_to_word, idx_to_tag, data))
            data = []
        idx += 1
    fo.close()
    if len(data):
        result.extend(run_model(model, idx_to_word, idx_to_tag, data))
    if lb:
        return result
    else:
        print()
        for x, _, y in result:
            print((x, y))

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model word_to_idx tag_to_idx test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    with torch.no_grad():
        predict()
