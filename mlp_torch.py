"""Multilayer Perceptron"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from numpy.typing import NDArray


class Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.h1 = nn.Linear(28*28, 64)
        self.o = nn.Linear(64, 10)

    def forward(self, p: torch.Tensor):
        ret = torch.relu(self.h1(p))
        ret = self.o(ret)
        return ret

net = Mlp()


### init

def init(model: nn.Module):
    if isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight)
        nn.init.zeros_(model.bias)

net.apply(init)

optimizer = Adam(net.parameters(), lr=1e-3, betas=(0.01, 0.1))
cost_fn = nn.CrossEntropyLoss(reduction='mean')


def train(data: NDArray, labels: NDArray, epochs: int, batch_size: int):

    train_dataset = TensorDataset(
        torch.from_numpy(data).float(),
        torch.from_numpy(labels)
    )
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):

        for batch_data, batch_labels in loader:

            optimizer.zero_grad()

            output = net(batch_data)
            loss = cost_fn(output, batch_labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            test_output = net(train_dataset.tensors[0])
            loss = cost_fn(test_output, train_dataset.tensors[1])
            acc = accuracy(test_output, train_dataset.tensors[1])
            print(f"Epoch: {epoch + 1} | loss: {loss:.6f}, acc: {acc*100:.4f}%")


### acc

def accuracy(output: torch.Tensor, labels: torch.Tensor):
    sums = torch.sum(torch.argmax(output, dim=1) == labels)
    return sums / labels.shape[0]


if __name__ == "__main__":

    ### get data
    train_data = np.load("./mnist/train_data.npy")
    train_data = train_data.reshape(60000, -1)
    train_labels = np.load("./mnist/train_labels.npy")

    test_data = np.load("./mnist/test_data.npy")
    test_data = test_data.reshape(10000, -1)
    test_labels = np.load("./mnist/test_labels.npy")

    train(train_data, train_labels, 50, 5400)

    test_output = net(torch.from_numpy(test_data).float())
    loss = cost_fn(test_output, torch.from_numpy(test_labels))
    acc = accuracy(test_output, torch.from_numpy(test_labels))
    print(f"Test | loss: {loss:.6f}, acc: {acc*100:.4f}%")
