import pdb
import os
import pandas as pd

from datareader import *
from DeepFM import *

if __name__ == '__main__':
    traindata = pd.read_csv("data/train.csv")
    testdata = pd.read_csv("data/test.csv")

    fd = featuredictionary(traindata, testdata)
    data_parser = dataparser(fd)
    Xi_train, Xv_train, Y_train = data_parser.parse(traindata, True)

    model = DeepFM(fd.feat_dim)
    model.train(Xi_train, Xv_train, Y_train)

