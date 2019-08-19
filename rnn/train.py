from model import *
from utils import *
from evaluate import *

def load_data():
    data = []
    bxc = [] # character sequence batch
    bxw = [] # word sequence batch
    by = [] # label batch
    cti = load_tkn_to_idx(sys.argv[2]) # char_to_idx
    wti = load_tkn_to_idx(sys.argv[3]) # word_to_idx
    itt = load_idx_to_tkn(sys.argv[4]) # idx_to_tag
    itw = idx_to_tkn(wti) # idx_to_word
    print("loading data...")
    fo = open(sys.argv[5], "r")
    for line in fo:
        line = line.strip()
        *x, y = [x.split(":") for x in line.split(" ")]
        xc, xw = zip(*[(list(map(int, xc.split("+"))), int(xw)) for xc, xw in x])
        bxc.append(xc)
        bxw.append(xw)
        by.append(int(y[0]))
        if len(by) == BATCH_SIZE:
            bxc, bxw = batchify(bxc, bxw, True, True)
            data.append((bxc, bxw, LongTensor(by)))
            bxc = []
            bxw = []
            by = []
    fo.close()
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, cti, wti, itt, itw

def train():
    num_epochs = int(sys.argv[-1])
    data, cti, wti, itt, itw = load_data()
    model = rnn(len(cti), len(wti), len(itt))
    print(model)
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    epoch = load_checkpoint(sys.argv[1], model) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        ii = 0
        loss_sum = 0
        timer = time()
        for xc, xw, y in data:
            ii += 1
            model.zero_grad()
            mask = maskset(xw)
            loss = F.nll_loss(model(xc, xw, mask), y)
            loss.backward()
            optim.step()
            loss_sum += loss.item()
        timer = time() - timer
        loss_sum /= len(data)
        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, model, ei, loss_sum, timer)
        if EVAL_EVERY and (ei % EVAL_EVERY == 0 or ei == epoch + num_epochs):
            args = [model, cti, wti, itt, itw]
            evaluate(predict(sys.argv[6], *args), True)
            model.train()
            print()
if __name__ == "__main__":
    if len(sys.argv) not in [7, 8]:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx training_data (validation_data) num_epoch" % sys.argv[0])
    print("cuda: %s" % CUDA)
    train()
