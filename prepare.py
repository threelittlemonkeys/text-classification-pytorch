import sys
from utils import *
from parameters import *

def load_data():
    data = []
    char_to_idx = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
    word_to_idx = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
    tag_to_idx = {}
    fo = open(sys.argv[1])
    for line in fo:
        x, y = line.split("\t")
        x = tokenize(x, "word")
        y = y.strip()
        for w in x:
            for c in w:
                if c not in char_to_idx:
                    char_to_idx[c] = len(char_to_idx)
            if w not in word_to_idx:
                word_to_idx[w] = len(word_to_idx)
        if y not in tag_to_idx:
            tag_to_idx[y] = len(tag_to_idx)
        data.append([str(word_to_idx[w]) for w in x] + [str(tag_to_idx[y])])
    data.sort(key = len, reverse = True)
    fo.close()
    return data, char_to_idx, word_to_idx, tag_to_idx

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, char_to_idx, word_to_idx, tag_to_idx = load_data()
    save_data(sys.argv[1] + ".csv", data)
    save_tkn_to_idx(sys.argv[1] + ".char_to_idx", char_to_idx)
    save_tkn_to_idx(sys.argv[1] + ".word_to_idx", word_to_idx)
    save_tkn_to_idx(sys.argv[1] + ".tag_to_idx", tag_to_idx)
