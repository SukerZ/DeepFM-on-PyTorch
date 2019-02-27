import pdb
import pandas as pd
import os
import csv

from datareader import *
from DeepFM import *

if __name__ == '__main__':
    traindata = pd.read_csv("data/train.csv")
    testdata = pd.read_csv("data/test.csv")

    fd = featuredictionary(traindata, testdata)
    data_parser = dataparser(fd)
    Xi_test, Xv_test, ID = data_parser.parse(testdata, False)
    model = DeepFM(fd.feat_dim)

    out = open('data/Submitt.csv', 'w', newline="")
    writer = csv.writer(out, dialect='excel')
    taitou = ["id", "target"]
    writer.writerow(taitou)

    size = 150; start = 0
    end = start + size
    while end <= len(Xi_test):
        Y_pre = model.test(Xi_test[start:end], Xv_test[start:end] )
        for i in range(Y_pre.size()[0]):
            line = [ ID[start + i], Y_pre[i].item() ]
            writer.writerow(line)

        if end >= len(Xi_test):
            end += 2
        else:
            start = end
            end = min( start + size, len(Xi_test) )