import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable

def eminput(x):
    return Variable(torch.tensor(x) )

class Network(nn.Module):
    def __init__(self, feature_size, embedding_size):
        super(Network, self).__init__()
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(feature_size, embedding_size)
        self.logistic = nn.Sequential(
            nn.Linear(embedding_size, 1)
        )
        self.n_hidden = 32; self.p = 0.5
        self.fc1 = nn.Sequential(
            nn.Linear(embedding_size, self.n_hidden),
            nn.Dropout(self.p), nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.Dropout(self.p), nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(self.n_hidden + self.embedding_size + 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        flag = True
        for i in range(self.feature_size):
            tmp = self.embedding(eminput(i) ).reshape(self.embedding_size, 1)
            if flag:
                weights = tmp
                flag = False
            else:
                weights = torch.cat((weights, tmp), 1)

        feature = x.clone().detach().reshape(self.feature_size, 1)
        embfeature = weights.mm(feature).reshape(1, self.embedding_size)
        logistic = self.logistic(embfeature)

        sum_squ = torch.zeros(self.embedding_size )
        for i in range(self.embedding_size):
            sum_squ[i] = embfeature[0][i] * embfeature[0][i]

        newfeature = torch.zeros(feature.size() )
        for j in range(self.feature_size):
            newfeature[j] = feature[j] * feature[j]

        newweights = torch.zeros(weights.size() )
        for i in range(self.embedding_size):
            for j in range(self.feature_size):
                newweights[i,j] = weights[i,j] * weights[i,j]

        squ_sum = newweights.mm(newfeature).reshape(1, self.embedding_size)
        fm = 0.5 * (sum_squ - squ_sum)
        ann = self.fc2(self.fc1(embfeature) )
        concat = torch.cat((ann[0], fm[0]), 0)
        concat = torch.cat((concat, logistic[0]), 0)

        return self.out(concat)
