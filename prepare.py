import sys
import re
from model import *
from utils import *

def load_data():
    data = []
    word_to_idx = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
    tag_to_idx = {}
    fo = open(sys.argv[1])
    for line in fo:
        x, y = line.split("\t")
        x = tokenize(x, UNIT)
        y = y.strip()
        for w in x:
            if w not in word_to_idx:
                word_to_idx[w] = len(word_to_idx)
        if y not in tag_to_idx:
            tag_to_idx[y] = len(tag_to_idx)
        data.append([str(word_to_idx[w]) for w in x] + [str(tag_to_idx[y])])
    data.sort(key = len, reverse = True)
    fo.close()
    return data, word_to_idx, tag_to_idx

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, word_to_idx, tag_to_idx = load_data()
    save_data(sys.argv[1] + ".tsv", data)
    save_tkn_to_idx(sys.argv[1] + ".word_to_idx", word_to_idx)
    save_tkn_to_idx(sys.argv[1] + ".tag_to_idx", tag_to_idx)
