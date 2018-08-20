import sys
import json
import re
import time
from collections import defaultdict

class rule():
    def __init__(self, label, cond, pt, weight = 1.):
        self.label = label
        self.cond = cond
        self.pt = pt
        self.re = re.compile(pt)
        self.weight = weight

class classifer(): # weighted majority vote classifier
    def __init__(self, fn_dict, fn_rules, verbose = False):
        self.dict = {}
        self.rules = []
        self.labels = {}
        self.verbose = verbose
        self.load_dict(fn_dict)
        self.load_rules(fn_rules)

    def load_dict(self, fn):
        fo = open(fn)
        for line in fo:
            tokens = line.split()
            for k in tokens[1:]:
                if k not in self.dict:
                    self.dict[k] = []
                self.dict[k].append(tokens[0])
        fo.close()
        for k, v in self.dict.items():
            self.dict[k] = "(" + "|".join(v) + ")"

    def load_rules(self, fn):
        count = 0
        fo = open(fn)
        for line in fo:
            count += 1
            if line[0] == "#": # comment line
                continue
            m = re.match("([A-Z]+(\|[A-Z]+)*) (\*|[_A-Z]+(\|[_A-Z]+)*) (.+)", line)
            if not m:
                sys.exit("Rule syntax error at line %d" % count)
            label = set(m.group(1).split("|"))
            cond = set(m.group(3).split("|"))
            pt = m.group(5)
            m = re.search(r"(?<!\\){([_A-Z]+)}", pt) # word class
            while m:
                k = m.group(1)
                if k in self.dict:
                    pt = pt[:m.start()] + self.dict[k] + pt[m.end():]
                m = re.search(r"(?<!\\){([_A-Z]+)}", pt)
            for y in label:
                if y not in self.labels:
                    self.labels[y] = len(self.labels)
            self.rules.append(rule(label, cond, pt))
        fo.close()

    def normalize(self, line):
        line = line.strip()
        line = line.lower()
        return line

    def predict(self, line):
        line = self.normalize(line)
        pred = defaultdict(float)
        pred["_"] = 0.
        if self.verbose:
            print(line)
        for i, rule in enumerate(self.rules):
            j = next(iter(rule.cond))
            if j == "_" and len(rule.cond) == 1 and len(pred) > 1:
                continue
            if j != "*" and not rule.cond.intersection(pred):
                continue
            for m in rule.re.finditer(line):
                if self.verbose:
                    print("rule[%d]: %s %s /%s/ \"%s\"" % (i, "|".join(rule.label), "|".join(rule.cond), rule.pt, m.group(0)))
                if j != "*" and rule.weight == 1:
                    pruned = filter(lambda x: x not in rule.label, list(pred))
                    for y in pruned:
                        del pred[y] # delete all the other labels
                for y in rule.label:
                    pred[y] += rule.weight

        # find the most voted label
        pred_sorted = sorted(pred.items(), key = lambda x: x[1], reverse = True)
        if self.verbose:
            print("%s\n" % ", ".join(["%s = %.2f" % (y, p) for y, p in pred_sorted]))

        return pred_sorted[0][0] # argmax

    def evaluate(self, fo):
        s = defaultdict(int) # entire set
        p = defaultdict(int) # positives
        t = defaultdict(int) # true positives
        for line in fo:
            line, value = line.split("\t")
            line = self.normalize(line)
            value = value.strip()
            pred = self.predict(line)
            s[value] += 1
            p[value] += 1
            if pred == value:
                t[value] += 1
        if not self.verbose:
            print()
        for y in s.keys():
            prec = t[y] / p[y]
            rec = t[y] / s[y]
            print("label = %s" % y)
            print("precision = %.2f (%d/%d)" % (prec, t[y], p[y]))
            print("recall = %.2f (%d/%d)" % (rec, t[y], s[y]))
            print("f1 = %.2f\n" % ((2 * prec * rec / (prec + rec)) if prec + rec else 0))

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s predict|evaluate dict rules data" % sys.argv[0])
    model = classifer(sys.argv[2], sys.argv[3], verbose = False)
    fin = open(sys.argv[4])
    if sys.argv[1] == "predict":
        fout = open(sys.argv[4] + ".out", "w")
        t = time.time()
        for line in fin:
            fout.write("%s\t%s\n" % (line.strip(), model.predict(line)))
        print("%f seconds" % (time.time() - t))
        fout.close()
    if sys.argv[1] == "evaluate":
        model.evaluate(fin)
    fin.close()
