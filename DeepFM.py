import pdb
import numpy as np
from time import time
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

from Network import *

class DeepFM(object):
    def save(self):
        print("Save model parameters.")
        torch.save(self.net.state_dict(), self.mn)

    def load(self):
        if os.path.exists(self.mn):
            print("Load model parameters.")
            self.net.load_state_dict(torch.load(self.mn))

    def __init__(self, feature_size):
        self.mn = "model.pth"
        self.embedding_size = 8
        self.net = Network(feature_size, self.embedding_size)
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.BCELoss()
        self.feature_size = feature_size
        self.batch_size = 128
        self.epoch = 10
        self.load()

    def train(self, Xi, Xv, Y):
        self.net.train()
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle(Xi, Xv, Y)
            total_batch = int(len(Y) / self.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, Y_batch = self.get_batch(Xi, Xv, Y, self.batch_size, i)
                Y_predict = self.predict(Xi_batch, Xv_batch)
                Y_batch = eminput(Y_batch).float()
                loss = self.loss_func(Y_predict, Y_batch )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print("Epoch", epoch + 1, "已完成。"); self.save()
        return

    def test(self, Xi, Xv):
        self.net.eval()
        return self.predict(Xi, Xv)

    def predict(self, Xi, Xv):
        n = len(Xv)
        input = torch.zeros((n, self.feature_size) )
        for i in range(len(Xi) ):
            for j in range(len(Xi[i])):
                input[i, Xi[i][j]] = Xv[i][j]

        Y_predict = torch.zeros((n,1) )
        for i in range(n):
            Y_predict[i] = self.net(input[i])

        return Y_predict

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    def shuffle(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)