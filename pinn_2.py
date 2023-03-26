import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from matplotlib import cm


# construct model
class Net(nn.Module):
    def __init__(self, NN):
        super(Net, self).__init__()

        self.hidden_l1 = nn.Linear(2, NN)
        self.hidden_l2 = nn.Linear(NN, NN//2)
        self.hidden_l3 = nn.Linear(NN//2, NN//4)
        self.hidden_l4 = nn.Linear(NN//4, NN//8)
        self.hidden_l5 = nn.Linear(NN//8, NN//16)
        self.output_layer = nn.Linear(NN//16, 1)

    def forward(self, x):
        out = 1/torch.cosh(self.hidden_l1(x))
        out = 1/torch.cosh(self.hidden_l2(out))
        out = 1/torch.cosh(self.hidden_l3(out))
        out = 1/torch.cosh(self.hidden_l4(out))
        out = 1/torch.cosh(self.hidden_l5(out))
        return self.output_layer(out)


def pde(x, net):
    u = net(x)
    grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True, allow_unused=True)[0]
    u_t = grad_u[:, 0:1]
    u_x = grad_u[:, 1:2]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True, allow_unused=True)[0][:, 1:2]

    return u_t - 0.1 * u_xx


net = Net(32)
mse_cost_function = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


# hyperparameters
iterations = 6000
Nb = 5000
Nf = 50


for epoch in range(iterations):
    optimizer.zero_grad()

    # loss of boundary conditions
    pt_i = np.zeros((Nb, 2))
    pt_i[:, 1] = np.random.uniform(-1, 1, Nb)
    pt_i = Variable(torch.from_numpy(pt_i).float(), requires_grad=False)
    pt_b = np.zeros((Nb * 2, 2))
    pt_b[:, 0] = np.random.uniform(0, 2, Nb * 2)
    pt_b[0:Nb, 1] = -1
    pt_b[Nb:, 1] = 1
    pt_b = Variable(torch.from_numpy(pt_b).float(), requires_grad=True)

    pt_b_out = net(pt_b)
    pt_u_x = torch.autograd.grad(pt_b_out, pt_b, grad_outputs=torch.ones_like(pt_b_out),
                                 create_graph=True, allow_unused=True)[0][:, 1:2]

    real_in = np.sign(pt_i[:, 1:2])
    mse_in = mse_cost_function(net(pt_i), real_in)
    mse_b = mse_cost_function(pt_u_x, torch.zeros((Nb * 2, 1)))

    # loss from PDE
    pt_f = np.zeros((Nf, 2))
    pt_f[:, 0] = np.random.uniform(0, 2, Nf)
    pt_f[:, 1] = np.random.uniform(-1, 1, Nf)
    pt_f = Variable(torch.from_numpy(pt_f).float(), requires_grad=True)

    mse_f = mse_cost_function(pde(pt_f, net), torch.zeros((Nf, 1)))

    # backward
    loss = 0.3*mse_b + 0.6*mse_in + 0.1*mse_f
    loss.backward()
    optimizer.step()

    with torch.autograd.no_grad():
        if (epoch + 1) % 300 == 0:
            print(f"Epoch: {epoch} | Loss: {loss.data}")


##################################################
## Draw the solution
##################################################

t = np.linspace(0, 2, 100)
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
