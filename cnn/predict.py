import sys
import re
from model import *
from utils import *

def load_model():
    cti = load_tkn_to_idx(sys.argv[2]) # char_to_idx
    wti = load_tkn_to_idx(sys.argv[3]) # word_to_idx
    itt = load_idx_to_tag(sys.argv[4]) # idx_to_tag
    model = cnn(len(cti), len(wti), len(itt))
    print(model)
    model.eval()
    load_checkpoint(sys.argv[1], model)
    return model, cti, wti, itt

def run_model(model, itt, batch):
    batch_size = len(batch) # real batch size
    while len(batch) < BATCH_SIZE:
        batch.append([-1, "", [], ""])
    batch.sort(key = lambda x: -len(x[2]))
    batch_len = max(len(batch[0][2]), max(KERNEL_SIZES))
    x = [[SOS_IDX] + x[2] + [EOS_IDX] + [PAD_IDX] * (batch_len - len(x[2])) for x in batch]
    result = model(LongTensor(x))
    for i in range(batch_size):
        y = itt[result[i].argmax()]
        p = round(max(result[i]).exp().item(), NUM_DIGITS)
        batch[i].append(y)
        batch[i].append(p)
        if VERBOSE:
            print()
            print(batch[i])
            y = torch.exp(result[i]).tolist()
            for j, p in sorted(enumerate(y), key = lambda x: -x[1]):
                print("%s %.6f" % (itt[j], p))
    return [(x[1], *x[3:]) for x in sorted(batch[:batch_size])]

def predict(filename, model, cti, wti, itt):
    bx = []
    data = []
    model, cti, wti, itt = load_model()
    fo = open(filename)
    for idx, line in enumerate(fo):
        line = line.strip()
        line, y = line.split("\t") if line.count("\t") else [line, None]
        x = tokenize(line, UNIT)
        x = [wti[w] if w in wti else UNK_IDX for w in x]
        bx.append(x)
        data.append([idx, line, y])
    fo.close()
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i + BATCH_SIZE]
        for y in run_model(model, itt, batch):
            yield y

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    with torch.no_grad():
        result = predict(sys.argv[5], *load_model())
        print()
        for x, y0, y1, p in result:
            print((x, y0, y1, p) if y0 else (x, y1, p))
