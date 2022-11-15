
from torchvision.datasets import MNIST
import numpy as np

trainset = MNIST("/home/flantori/Datasets", train=True, download=True)
train_data = trainset.data.numpy()
train_labels = trainset.targets.numpy()

testset = MNIST("/home/flantori/Datasets", train=False, download=True)
test_data = testset.data.numpy()
test_labels = testset.targets.numpy()
print(test_data.shape)

np.save("./mnist/train_data.npy", train_data)
np.save("./mnist/train_labels.npy", train_labels)
np.save("./mnist/test_data.npy", test_data)
np.save("./mnist/test_labels.npy", test_labels)
