import re
from model import *

NUM_DIGITS = 4

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

def idx_to_tkn(tkn_to_idx):
    return [x for x, _ in sorted(tkn_to_idx.items(), key = lambda x: x[1])]

def save_data(filename, data):
    fo = open(filename, "w")
    for seq in data:
        fo.write(" ".join(seq) + "\n")
    fo.close()

def load_tkn_to_idx(filename):
    print("loading %s" % filename)
    tkn_to_idx = {}
    fo = open(filename)
    for line in fo:
        line = line.strip()
        tkn_to_idx[line] = len(tkn_to_idx)
    fo.close()
    return tkn_to_idx

def load_idx_to_tkn(filename):
    print("loading %s" % filename)
    idx_to_tkn = []
    fo = open(filename)
    for line in fo:
        line = line.strip()
        idx_to_tkn.append(line)
    fo.close()
    return idx_to_tkn

def save_tkn_to_idx(filename, tkn_to_idx):
    fo = open(filename, "w")
    for tkn, _ in sorted(tkn_to_idx.items(), key = lambda x: x[1]):
        fo.write("%s\n" % tkn)
    fo.close()

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

def heatmap(m, x, idx_to_word):
    y = []
    y.append([idx_to_word[c] for c in x]) # input
    for v in m: # weights
        y.append([x for x in v[:len(x)]])
    return y

def mat2tsv(m, ch = True, rh = False, nd = NUM_DIGITS, delim ="\t"):
    f = "%%.%df" % nd
    if ch: # column header
        tsv = delim.join([x for x in m[0]]) + "\n" # source sequence
    for row in m[ch:]:
        if rh: # row header
            tsv += row[0] + delim # target sequence
        tsv += delim.join([f % x for x in row[rh:]]) + "\n"
    return tsv

def f1(p, r):
    if p + r:
        return 2 * p * r / (p + r)
    return 0
