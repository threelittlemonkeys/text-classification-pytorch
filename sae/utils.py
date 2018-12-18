import re
from model import *

def normalize(x):
    # x = re.sub("[^ a-zA-Z0-9\uAC00-\uD7A3]+", " ", x)
    x = re.sub("\s+", " ", x)
    x = re.sub("^ | $", "", x)
    x = x.lower()
    return x

def tokenize(x, unit):
    x = normalize(x)
    if unit == "char":
        x = re.sub(" ", "", x)
        return list(x)
    if unit == "word":
        return x.split(" ")
    return x

def load_tag_to_idx(filename):
    print("loading tag_to_idx...")
    tag_to_idx = {}
    fo = open(filename)
    for line in fo:
        line = line.strip()
        tag_to_idx[line] = len(tag_to_idx)
    fo.close()
    return tag_to_idx

def load_word_to_idx(filename):
    print("loading word_to_idx...")
    word_to_idx = {}
    fo = open(filename)
    for line in fo:
        line = line.strip()
        word_to_idx[line] = len(word_to_idx)
    fo.close()
    return word_to_idx

def load_checkpoint(filename, model = None):
    print("loading model...")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch

def save_checkpoint(filename, model, epoch, loss, time):
    print("epoch = %d, loss = %f, time = %f" % (epoch, loss, time))
    if filename and model:
        print("saving model...")
        checkpoint = {}
        checkpoint["state_dict"] = model.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = loss
        torch.save(checkpoint, filename + ".epoch%d" % epoch)
        print("saved model at epoch %d" % epoch)

def f1(p, r):
    if p + r:
        return 2 * p * r / (p + r)
    return 0

def heatmap(m, x, idx_to_word, delim = "\t"):
    y = []
    y.append([idx_to_word[c] for c in x]) # input
    for v in m: # weights
        y.append([x for x in v[:len(x)]])
    return y

def mat2csv(m, delim ="\t", n = 0):
    k = 10 ** -n
    csv = delim.join([x for x in m[0]]) + "\n"
    for v in m[1:]:
        if n:
            csv += delim.join([str(round(x, n)) if x > k else "" for x in v]) + "\n"
        else:
            csv += delim.join([str(x) for x in v]) + "\n"
    return csv