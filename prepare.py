from utils import *

def load_data():
    data = []
    cti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX} # char_to_idx
    wti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX} # word_to_idx
    tti = {} # tag_to_idx
    fo = open(sys.argv[1])
    for line in fo:
        x, y = line.split("\t")
        x = tokenize(x, UNIT)
        y = y.strip()
        for w in x:
            for c in w:
                if c not in cti:
                    cti[c] = len(cti)
            if w not in wti:
                wti[w] = len(wti)
        if y not in tti:
            tti[y] = len(tti)
        x = ["+".join(str(cti[c]) for c in w) + ":%d" % wti[w] for w in x]
        y = [str(tti[y])]
        data.append(x + y)
    fo.close()
    data.sort(key = len, reverse = True)
    return data, cti, wti, tti

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, cti, wti, tti = load_data()
    save_data(sys.argv[1] + ".csv", data)
    save_tkn_to_idx(sys.argv[1] + ".char_to_idx", cti)
    save_tkn_to_idx(sys.argv[1] + ".word_to_idx", wti)
    save_tkn_to_idx(sys.argv[1] + ".tag_to_idx", tti)
