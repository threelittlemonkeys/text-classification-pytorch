import sys
import re
from model import *
from utils import *

def load_model():
    word_to_idx = load_word_to_idx(sys.argv[2])
    tag_to_idx = load_tag_to_idx(sys.argv[3])
    idx_to_word = [x for x, _ in sorted(word_to_idx.items(), key = lambda x: x[1])]
    idx_to_tag = [x for x, _ in sorted(tag_to_idx.items(), key = lambda x: x[1])]
    model = rnn(len(word_to_idx), len(tag_to_idx))
    print(model)
    model.eval()
    load_checkpoint(sys.argv[1], model)
    return model, word_to_idx, tag_to_idx, idx_to_word, idx_to_tag

def run_model(model, idx_to_word, idx_to_tag, batch):
    batch_size = len(batch) # real batch size
    while len(batch) < BATCH_SIZE:
        batch.append([-1, "", [UNK_IDX], ""])
    batch.sort(key = lambda x: -len(x[2]))
    batch_len = len(batch[0][2])
    x = LongTensor([x[2] + [PAD_IDX] * (batch_len - len(x[2])) for x in batch])
    result = model(x, maskset(x))
    if VERBOSE:
        Va = model.attn.Va.squeeze(2).tolist() # attention weights
    for i in range(batch_size):
        y = idx_to_tag[argmax(result[i])]
        batch[i].append(y)
        if VERBOSE:
            print(batch[i][1])
            y = enumerate(result[i].exp().tolist())
            for a, b in sorted(y, key = lambda x: -x[1]):
                print(idx_to_tag[a], round(b, 4))
            print(mat2csv(heatmap(Va[i], batch[i][2], idx_to_word))) # attention heatmap
    return [(x[1], *x[3:]) for x in sorted(batch[:batch_size])]

def predict(lb = False):
    data = []
    result = []
    model, word_to_idx, tag_to_idx, idx_to_word, idx_to_tag = load_model()
    fo = open(sys.argv[4])
    for idx, line in enumerate(fo):
        line = line.strip()
        line, y = line.split("\t") if lb else [line, ""]
        x = tokenize(line, UNIT)
        x = [word_to_idx[i] if i in word_to_idx else UNK_IDX for i in x]
        data.append([idx, line, x, y])
    fo.close()
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i + BATCH_SIZE]
        result.extend(run_model(model, idx_to_word, idx_to_tag, batch))
    if lb:
        return result
    for x, _, y in result:
        print((x, y))

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model word_to_idx tag_to_idx test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    with torch.no_grad():
        predict()
