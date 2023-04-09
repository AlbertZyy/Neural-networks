
import torch
import torch.nn as nn
from fealpy.pinn.modules import GradAttention
from fealpy.pinn.sampler import ISampler


def df(p):
    return 2*p


s = ISampler(100, [[0, 1], [0, 1]], requires_grad=False)

q = s.run()

net = GradAttention(df)

q2 = net(q)
print(q2)

from matplotlib import pyplot as plt

plt.scatter(q[..., 0], q[..., 1])
plt.scatter(q2[..., 0], q2[..., 1])
plt.show()
