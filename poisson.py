from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


pde_data = CosCosData()

NN: int = 64
pinn = nn.Sequential(
    nn.Linear(2, NN),
    nn.Tanh(),
    nn.Linear(NN, NN//2),
    nn.Tanh(),
    nn.Linear(NN//2, NN//4),
    nn.Tanh(),
    nn.Linear(NN//4, 1)
)

def pde(func, x):
    u = func(x)
    grad_u = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, allow_unused=True
    )[0]

    u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]

    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x),
        create_graph=True, allow_unused=True
    )[0][:, 0:1]

    u_yy = torch.autograd.grad(
        u_y, x, grad_outputs=torch.ones_like(u_x),
        create_graph=True, allow_unused=True
    )[0][:, 1:2]

    return u_xx + u_yy + np.pi**2 * u

mse_cost_func = nn.MSELoss(reduction='mean')
optim = torch.optim.Adam(pinn.parameters(), lr=0.001, weight_decay=0)


# hyperparameters
iterations = 3000
Nb = 3000
Nf = 300


for epoch in range(iterations):
    optim.zero_grad()

    # loss of boundary conditions
    pt_b = np.zeros((4*Nb, 2))
    pt_b[0:Nb, 0] = np.random.uniform(0, 1, Nb)
    pt_b[Nb:2*Nb, 0] = 1
    pt_b[Nb:2*Nb, 1] = np.random.uniform(0, 1, Nb)
    pt_b[2*Nb:3*Nb, 0] = np.random.uniform(0, 1, Nb)
    pt_b[2*Nb:3*Nb, 1] = 1
    pt_b[3*Nb:, 1] = np.random.uniform(0, 1, Nb)
    pt_b = Variable(torch.from_numpy(pt_b).float(), requires_grad=False)

    pt_b_out = pinn(pt_b)
    real_b = pde_data.dirichlet(pt_b).unsqueeze(-1)
    mse_b = mse_cost_func(pt_b_out, real_b)

    # loss from PDE
    pt_f = np.zeros((Nf, 2))
    pt_f[:, 0] = np.random.uniform(0, 1, Nf)
    pt_f[:, 1] = np.random.uniform(0, 1, Nf)
    pt_f = Variable(torch.from_numpy(pt_f).float(), requires_grad=True)

    mse_f = mse_cost_func(pde(pinn, pt_f), torch.zeros((Nf, 1)))

    # backward
    loss = 0.9*mse_b + 0.1*mse_f
    loss.backward()
    optim.step()

    with torch.autograd.no_grad():
        for params in pinn.parameters():
            params -= 1e-3 * params**3/(params**2 + 3**2)
        if (epoch + 1) % 300 == 0:
            print(f"Epoch: {epoch} | Loss: {loss.data}")


@cartesian
def f(p: np.ndarray):
    pt = Variable(torch.from_numpy(p).float(), requires_grad=True)
    val = pinn(pt).detach().numpy().squeeze(-1)
    real = pde_data.solution(p)
    return (val - real) ** 2


domain = [0, 1, 0, 1]
mesh = MF.boxmesh2d(domain, nx=10, ny=10, meshtype='tri')
space = LagrangeFiniteElementSpace(mesh, 1)

error = np.sqrt(space.integralalg.cell_integral(f).sum())
print(f"L2 error: {error}")


from matplotlib import pyplot as plt
from matplotlib import cm

x = np.linspace(0, 1, 30)
y = np.linspace(0, 1, 30)
X, Y = np.meshgrid(x, y)
x = np.ravel(X).reshape(-1, 1)
y = np.ravel(Y).reshape(-1, 1)
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True)
pt_u0 = pinn(torch.cat([pt_x, pt_y], 1))
u_plot = pt_u0.data.cpu().numpy()

pt_u0 = u_plot.reshape(30, 30)

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1, projection='3d')
axes.set_zlim([-1, 1])
axes.plot_surface(X, Y, pt_u0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
axes.set_xlabel('t')
axes.set_ylabel('x')
axes.set_zlabel('u')
plt.show()
