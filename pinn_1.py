import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from matplotlib import cm


# 模型搭建
class Net(nn.Module):
    def __init__(self, NN):
        super(Net, self).__init__()

        self.hidden_l1 = nn.Linear(2, NN)
        self.hidden_l2 = nn.Linear(NN, int(NN/2))
        self.hidden_l3 = nn.Linear(int(NN/2), int(NN/4))
        self.output_layer = nn.Linear(int(NN/4), 1)

    def forward(self, x):
        out = 1 - torch.tanh(self.hidden_l1(x))**2
        out = 1 - torch.tanh(self.hidden_l2(out))**2
        out = 1 - torch.tanh(self.hidden_l3(out))**2
        return self.output_layer(out)


def pde(x, net):
    u = net(x)
    grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True, allow_unused=True)[0]
    d_t = grad_u[:, 0:1]
    d_x = grad_u[:, 1:2]

    u_xx = torch.autograd.grad(d_x, x, grad_outputs=torch.ones_like(d_x),
                               create_graph=True, allow_unused=True)[0][:,1:2]

    w = torch.tensor(0.01 / np.pi)
    return d_t + u * d_x - w * u_xx  # 公式（1）


net = Net(32)
mse_cost_function = torch.nn.MSELoss(reduction='mean')  # Mean squared error
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


# 超参数
iterations = 6000
Nb = 3000
Nf = 30


for epoch in range(iterations):
    optimizer.zero_grad()

    # 以下求边界条件的损失
    pt_b = np.zeros((Nb * 3, 2))
    pt_b[:Nb, 1] = np.random.uniform(-1, 1, Nb)
    pt_b[Nb:, 0] = np.random.uniform(0, 1, Nb * 2)
    pt_b[Nb:2*Nb, 1] = -1
    pt_b[2*Nb:, 1] = 1
    pt_b = Variable(torch.from_numpy(pt_b).float(), requires_grad=False)

    real_b = np.zeros((Nb * 3, 1))
    real_b[:Nb, 0] = -np.sin(np.pi * pt_b[:Nb, 1])
    real_b = Variable(torch.from_numpy(real_b).float(), requires_grad=False)

    mse_b = mse_cost_function(net(pt_b), real_b)

    # 以下求 PDE 的损失
    pt_f = np.zeros((Nf, 2))
    pt_f[:, 0] = np.random.uniform(0, 1, Nf)
    pt_f[:, 1] = np.random.uniform(-1, 1, Nf)
    pt_f = Variable(torch.from_numpy(pt_f).float(), requires_grad=True)

    mse_f = mse_cost_function(pde(pt_f, net), torch.zeros((Nf, 1)))

    # 反向传播
    loss = mse_b + 0.1*mse_f
    loss.backward()
    optimizer.step()

    with torch.autograd.no_grad():
        if epoch % 300 == 0:
            print(epoch, "Traning Loss:", loss.data)


## 画图 ##

t = np.linspace(0, 1, 100)
x = np.linspace(-1, 1, 256)
ms_t, ms_x = np.meshgrid(t, x)
x = np.ravel(ms_x).reshape(-1, 1)
t = np.ravel(ms_t).reshape(-1, 1)
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True)
pt_u0 = net(torch.cat([pt_t, pt_x], 1))
u = pt_u0.data.cpu().numpy()

pt_u0 = u.reshape(256, 100)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_zlim([-1, 1])
ax.plot_surface(ms_t, ms_x, pt_u0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u')
plt.show()
#plt.savefig('Preddata.png')
#plt.close(fig)
