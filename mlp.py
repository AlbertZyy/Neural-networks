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

grad_w_squares: _opt_arrs = [None] * len(features)
grad_b_squares: _opt_arrs = [None] * len(features)


### init

for i in range(1, len(features)):
    bl = np.zeros((1, features[i]), dtype=np.float64)
    wl = np.random.randn(features[i-1], features[i]) * 0.1
    wl *= sqrt(2 / (features[i-1] + features[i]))
    bias[i] = bl
    weights[i] = wl
    grad_w_squares[i] = np.zeros((features[i-1], features[i]), dtype=np.float64)
    grad_b_squares[i] = np.zeros((1, features[i]), dtype=np.float64)


def init_squares():
    for i in range(1, len(features)):
        grad_w_squares[i] *= 0
        grad_b_squares[i] *= 0


### activation and cost functions

def relu(x: NDArray):
    ret = x.copy()
    ret[x < 0] *= 0.001
    return ret


def g_relu(x: NDArray):
    ret = np.ones_like(x) * 0.001
    ret[x > 0] = 1.0
    return ret


def softmax(x: NDArray):
    max_val = np.max(x, axis=1, keepdims=True)
    expx = np.exp(x - max_val)
    summary = np.sum(expx, axis=1, keepdims=True)
    return expx / summary


def loss_fn(o: NDArray, labels: NDArray):
    ret = o[np.arange(o.shape[0]), labels]
    np.log(ret, out=ret)
    return -np.average(ret)


def g_entropy_softmax(o: NDArray, labels: NDArray):
    one_hot = np.zeros_like(o)
    one_hot[np.arange(o.shape[0]), labels] = 1.0
    ret = o - one_hot
    return ret / o.shape[0]


### forward and backward functions

def forward(x: NDArray):
    outputs[0] = x
    for i in range(1, len(features)):
        zi = outputs[i-1]@weights[i] + bias[i]
        if i == len(features) - 1:
            ai = softmax(zi)
        else:
            ai = relu(zi)
        outputs[i], zs[i] = ai, zi


def backward(grad_out: NDArray):
    grad_outs[-1] = grad_out
    for i in range(len(features)-1, 0, -1):
        if i == len(features) - 1:
            jz = grad_outs[i]
        else:
            jz = grad_outs[i] * g_relu(zs[i])

        grad_w[i] = outputs[i-1].T@jz
        grad_b[i] = np.sum(jz, axis=0, keepdims=True)
        grad_outs[i-1] = jz@weights[i].T


### optimizer

def step(lr: float):
    for i in range(1, len(features)):
        grad_w_squares[i] += grad_w[i]**2
        grad_b_squares[i] += grad_b[i]**2
        weights[i] -= lr/np.sqrt(grad_w_squares[i] + 1) * grad_w[i]
        bias[i] -= lr/np.sqrt(grad_b_squares[i] + 1) * grad_b[i]


### training

def dataloader(data: NDArray, labels: NDArray, batch_size: int, rate: float=0.8):
    length = data.shape[0]
    idx = np.random.choice(length, int(length*rate))
    data = data[idx, ...]
    labels = labels[idx, ...]
    for i in range(ceil(length / batch_size)):
        if i + batch_size >= length:
            yield data[i:, ...], labels[i:]
        else:
            yield data[i: i + batch_size, ...], labels[i: i + batch_size]


def train(data: NDArray, labels: NDArray, lr: float, epochs: int, batch_size: int):
    init_squares()
    for epoch in range(epochs):

        for batch_data, batch_labels in dataloader(data, labels, batch_size):

            forward(batch_data)
            g_loss = g_entropy_softmax(outputs[-1], batch_labels)
            backward(g_loss)
            step(lr)

        if epoch % 5 == 0:

            loss = loss_fn(outputs[-1], batch_labels)
            acc = accuracy(outputs[-1], batch_labels)
            print(f"Epoch: {epoch} | loss: {loss:.4f}, acc: {acc*100:.4f}%")


### acc

def accuracy(output: NDArray, labels: NDArray):
    sums = np.sum(np.argmax(output, axis=1) == labels)
    return sums / labels.shape[0]


if __name__ == "__main__":

    ### get data
    train_data: NDArray = np.load("./mnist/train_data.npy")
    train_data = train_data.reshape(60000, -1)
    train_labels = np.load("./mnist/train_labels.npy")
    test_data = np.load("./mnist/test_data.npy")
    test_data = test_data.reshape(60000, -1)
    test_labels = np.load("./mnist/test_labels.npy")

    train(train_data, train_labels, 0.001, 120, 5000)

    forward(test_data)
    loss = loss_fn(outputs[-1], test_labels)
    acc = accuracy(outputs[-1], test_labels)
    print(f"Test | loss: {loss:.4f}, acc: {acc*100:.4f}%")
