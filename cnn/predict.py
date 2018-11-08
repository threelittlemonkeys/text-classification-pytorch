import sys
import re
from model import *
from utils import *

def load_model():
    word_to_idx = load_word_to_idx(sys.argv[2])
    tag_to_idx = load_tag_to_idx(sys.argv[3])
    idx_to_tag = [tag for tag, _ in sorted(tag_to_idx.items(), key = lambda x: x[1])]
    model = cnn(len(word_to_idx), len(tag_to_idx))
    print(model)
    model.eval()
    load_checkpoint(sys.argv[1], model)
    return model, word_to_idx, tag_to_idx, idx_to_tag

def run_model(model, idx_to_tag, data):
    batch = []
    z = len(data)
    while len(data) < BATCH_SIZE:
        data.append(["", [], ""])
    data.sort(key = lambda x: len(x[1]), reverse = True)
    batch_len = len(data[0][1])
    batch = [[SOS_IDX] + x[1] + [EOS_IDX] + [PAD_IDX] * (batch_len - len(x[1])) for x in data]
    result = model(LongTensor(batch))
    for i in range(z):
        data[i] = (data[i][0], data[i][2], idx_to_tag[result[i].argmax()])
        if VERBOSE:
            print()
            print(data[i])
            y = torch.exp(result[i]).tolist()
            for j, p in sorted(enumerate(y), key = lambda x: x[1], reverse = True):
                print("%s %.6f" % (idx_to_tag[j], p))
    return data[:z]

def predict(lb = False):
    data = []
    result = []
    model, word_to_idx, tag_to_idx, idx_to_tag = load_model()
    fo = open(sys.argv[4])
    for line in fo:
        line = line.strip()
        line, y = line.split("\t") if lb else [line, ""]
        x = tokenize(line, UNIT)
        x = [word_to_idx[i] if i in word_to_idx else UNK_IDX for i in x]
        data.append([line, x, y])
        if len(data) == BATCH_SIZE:
            result.extend(run_model(model, idx_to_tag, data))
            data = []
    fo.close()
    if len(data):
        result.extend(run_model(model, idx_to_tag, data))
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
    predict()
