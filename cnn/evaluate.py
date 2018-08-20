import sys
import re
from model import *
from utils import *
from collections import defaultdict

def load_model():
    word_to_idx = load_word_to_idx(sys.argv[2])
    tag_to_idx = load_tag_to_idx(sys.argv[3])
    idx_to_tag = [tag for tag, _ in sorted(tag_to_idx.items(), key = lambda x: x[1])]
    model = word_cnn(len(word_to_idx), len(tag_to_idx))
    model.eval()
    if CUDA:
        model = model.cuda()
    print(model)
    load_checkpoint(sys.argv[1], model)
    return model, word_to_idx, tag_to_idx, idx_to_tag

def run_model(model, idx_to_tag, data):
    batch = []
    z = len(data)
    while len(data) < BATCH_SIZE:
        data.append(["", [], ""])
    data.sort(key = lambda x: len(x[1]), reverse = True)
    batch_len = len(data[0][1])
    batch = [x[1] + [PAD_IDX] * (batch_len - len(x[1])) for x in data]
    result = model(LongTensor(batch))
    for i in range(z):
        m = argmax(result[i])
        y1 = idx_to_tag[m]
        p = scalar(torch.exp(result[i][m]))
        data[i].extend([y1, p])
        if VERBOSE:
            x = data[i][0]
            y0 = data[i][2]
            print("\t".join([x, y0, y1, str(round(p, 6))]))
    return data[:z]

def predict():
    data = []
    result = []
    model, word_to_idx, tag_to_idx, idx_to_tag = load_model()
    fo = open(sys.argv[4])
    for line in fo:
        line, y0 = line.split("\t")
        x = tokenize(line, "char")
        y0 = y0.strip()
        x = [word_to_idx[i] for i in x if i in word_to_idx]
        data.append([line, x, y0])
        if len(data) == BATCH_SIZE:
            result.extend(run_model(model, idx_to_tag, data))
            data = []
    fo.close()
    if len(data):
        result.extend(run_model(model, idx_to_tag, data))
    evaluate(result)

def evaluate(result):
    s = defaultdict(int) # entire set
    p = defaultdict(int) # positive
    t = defaultdict(int) # true positive
    for x in result:
        y0 = x[2] # actual value
        y1 = x[3] # predicted outcome
        s[y0] += 1
        p[y1] += 1
        if y0 == y1:
            t[y0] += 1
    for y in s.keys():
        prec = t[y] / p[y] if p[y] else 0
        rec = t[y] / s[y]
        print("\nlabel = %s" % y)
        print("precision = %.2f (%d/%d)" % (prec, t[y], p[y]))
        print("recall = %.2f (%d/%d)" % (rec, t[y], s[y]))
        print("f1 = %.2f" % ((2 * prec * rec / (prec + rec)) if prec + rec else 0))

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model word_to_idx tag_to_idx test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    predict()
