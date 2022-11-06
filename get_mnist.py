
from torchvision.datasets import MNIST
import numpy as np

data = MNIST("/home/flantori/Datasets", train=True, download=True)
train_data = data.train_data.numpy()
train_labels = data.train_labels.numpy()
test_data = data.test_data.numpy()
test_labels = data.test_labels.numpy()
print(train_data.shape)

np.save("./mnist/train_data.npy", train_data)
np.save("./mnist/train_labels.npy", train_labels)
np.save("./mnist/test_data.npy", test_data)
np.save("./mnist/test_labels.npy", test_labels)
