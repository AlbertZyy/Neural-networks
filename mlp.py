"""Multilayer Perceptron"""

from math import ceil, sqrt
from typing import Optional, List

import numpy as np
from numpy.typing import NDArray

_opt_arrs = List[Optional[NDArray]]


features = [28*28, 64, 10]

bias: _opt_arrs = [None] * len(features)
weights: _opt_arrs = [None] * len(features)
zs: _opt_arrs = [None] * len(features)
outputs: _opt_arrs = [None] * len(features)

grad_b: _opt_arrs = [None] * len(features)
grad_w: _opt_arrs = [None] * len(features)
grad_outs: _opt_arrs = [None] * len(features)


### init

for i in range(1, len(features)):
    bl = np.zeros((features[i], 1), dtype=np.float64)
    wl = np.random.randn(features[i], features[i-1]) * 0.1
    wl *= sqrt(2 / (features[i-1] + features[i]))
    bias[i] = bl
    weights[i] = wl


### activation and cost functions

def relu(x: NDArray):
    ret = np.array([x, 0.001*x])
    return np.max(ret, axis=0)


def g_relu(x: NDArray):
    ret = np.ones_like(x) * 0.001
    ret[x > 0] = 1.0
    return ret


def softmax(x: NDArray):
    expx = np.exp(x)
    summary = np.sum(expx, axis=0, keepdims=True)
    return expx / summary


def loss_fn(o: NDArray, labels: NDArray):
    ln_out = np.log(o)
    losses = ln_out[labels, np.arange(ln_out.shape[1])]
    return -np.average(losses)


def g_entropy_softmax(o: NDArray, labels: NDArray):
    one_hot = np.zeros_like(o)
    one_hot[labels, np.arange(one_hot.shape[1])] = 1.0
    ret = o - one_hot
    return ret / o.shape[1]


### dropout

def dropout_mask(p: float, shape):
    ret = np.random.rand(*shape)
    return ret > p

### forward and backward functions

def forward(x: NDArray, dropout: float = 0.0):
    outputs[0] = x
    for i in range(1, len(features)):
        zi = weights[i]@outputs[i-1] + bias[i]
        zi *= dropout_mask(dropout, zi.shape)/(1-dropout)
        if i == len(features) - 1:
            ai = softmax(zi)
        else:
            ai = relu(zi)
        outputs[i], zs[i] = ai, zi


def backward(grad_out: NDArray):
    grad_outs[-1] = grad_out
    for i in range(len(features)-1, 0, -1):
        jz = grad_outs[i] * g_relu(zs[i])
        jw = jz@outputs[i-1].T
        jb = np.sum(jz, axis=1, keepdims=True)
        ja = weights[i].T@jz
        grad_b[i], grad_w[i], grad_outs[i-1] = jb, jw, ja


### optimizer

def step(lr: float):
    for i in range(1, len(features)):
        weights[i] -= lr * grad_w[i]
        bias[i] -= lr * grad_b[i]


### training

EPOCHS = 100
BATCH_SIZE = 5000

def dataloader(data: NDArray, labels: NDArray, batch_size: int):
    length = data.shape[1]
    sdata = data.copy()
    for i in range(ceil(length / batch_size)):
        if i + batch_size >= length:
            yield sdata[:, i:], labels[i:]
        else:
            yield sdata[:, i: i + batch_size], labels[i: i + batch_size]


def train(data: NDArray, labels: NDArray):
    for epoch in range(EPOCHS):

        for batch_data, batch_labels in dataloader(data, labels, BATCH_SIZE):

            forward(batch_data, 0.1)
            g_loss = g_entropy_softmax(outputs[-1], batch_labels)
            backward(g_loss)
            step(0.001)

        if epoch % 5 == 0:
            loss = loss_fn(outputs[-1], batch_labels)
            forward(batch_data)
            acc = accuracy(outputs[-1], batch_labels)
            print(f"Epoch: {epoch} | loss: {loss:.4f}, acc: {acc*100:.4f}%")


### acc
def accuracy(output: NDArray, labels: NDArray):
    sums = np.sum(np.argmax(output, axis=0) == labels)
    return sums / labels.shape[0]


if __name__ == "__main__":

    ### get data
    train_data: NDArray = np.load("./mnist/train_data.npy")
    train_data = train_data.reshape(60000, -1).T
    train_labels = np.load("./mnist/train_labels.npy")
    test_data = np.load("./mnist/test_data.npy")
    test_data = test_data.reshape(60000, -1).T
    test_labels = np.load("./mnist/test_labels.npy")

    train(train_data, train_labels)

    forward(test_data)
    loss = loss_fn(outputs[-1], test_labels)
    acc = accuracy(outputs[-1], test_labels)
    print(f"Test | loss: {loss:.4f}, acc: {acc*100:.4f}%")
