import sys
import re
from model import *
from utils import *

def load_model():
    word_to_idx = load_tkn_to_idx(sys.argv[2])
    tag_to_idx = load_tkn_to_idx(sys.argv[3])
    idx_to_tag = idx_to_tkn(tag_to_idx)
    model = cnn(len(word_to_idx), len(tag_to_idx))
    print(model)
    model.eval()
    load_checkpoint(sys.argv[1], model)
    return model, word_to_idx, tag_to_idx, idx_to_tag

def run_model(model, idx_to_tag, batch):
    batch_size = len(batch) # real batch size
    while len(batch) < BATCH_SIZE:
        batch.append([-1, "", [], ""])
    batch.sort(key = lambda x: -len(x[2]))
    batch_len = max(len(batch[0][2]), max(KERNEL_SIZES))
    x = [[SOS_IDX] + x[2] + [EOS_IDX] + [PAD_IDX] * (batch_len - len(x[2])) for x in batch]
    result = model(LongTensor(x))
    for i in range(batch_size):
        y = idx_to_tag[result[i].argmax()]
        p = round(max(result[i]).exp().item(), NUM_DIGITS)
        batch[i].append(y)
        batch[i].append(p)
        if VERBOSE:
            print()
            print(batch[i])
            y = torch.exp(result[i]).tolist()
            for j, p in sorted(enumerate(y), key = lambda x: -x[1]):
                print("%s %.6f" % (idx_to_tag[j], p))
    return [(x[1], *x[3:]) for x in sorted(batch[:batch_size])]

def predict(filename, lb, model, word_to_idx, tag_to_idx, idx_to_tag):
    data = []
    model, word_to_idx, tag_to_idx, idx_to_tag = load_model()
    fo = open(filename)
    for idx, line in enumerate(fo):
        line = line.strip()
        line, y = line.split("\t") if lb else [line, ""]
        x = tokenize(line, UNIT)
        x = [word_to_idx[i] if i in word_to_idx else UNK_IDX for i in x]
        data.append([idx, line, x, y])
    fo.close()
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i + BATCH_SIZE]
        for y in run_model(model, idx_to_tag, batch):
            yield y

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model word_to_idx tag_to_idx test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    with torch.no_grad():
        result = predict(sys.argv[4], False, *load_model())
        print()
        for x, _, y, p in result:
            print((x, y, p))
