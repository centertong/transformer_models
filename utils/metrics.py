import numpy as np

class Accuracy(object):
    def __init__(self,):
        self.total_count = 0
        self.correct_count = 0

    def add(self, preds, labels):
        preds = preds.argmax(dim=-1)
        self.total_count += preds.shape[0]
        self.correct_count += (preds == labels).sum()

    def calc(self):
        if self.total_count == 0: return 0
        return self.correct_count / self.total_count


class MacroF1:
    def __init__(self, l2c):
        self.l2c = l2c
        self.table = np.zeros((len(self.l2c), len(self.l2c)))

    def add(self, preds, labels):
        for pred, label in zip(preds, labels):
            self.table[label, pred.argmax()] += 1

    def calc(self, mean=True):
        precision = self.table.diagonal() / (self.table.sum(axis=0) + 1e-15)
        recall = self.table.diagonal() / (self.table.sum(axis=1) + 1e-15)
        if mean:
            return (2 * precision * recall / (recall + precision + 1e-15)).mean()
        return (2 * precision * recall / (recall + precision + 1e-15))

class Pearson:
    def __init__(self):
        self.pred = []
        self.label = []

    def add(self, preds, labels):
        self.pred += preds
        self.label += labels

    def calc(self):
        return np.corrcoef(self.pred, self.label)