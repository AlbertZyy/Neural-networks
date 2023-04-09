
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
from fealpy.pinn.machine import Solution, LearningMachine
from fealpy.pinn.grad import grad_by_fts
from fealpy.pinn.sampler import ISampler, BoxBoundarySampler


net = nn.Sequential(
    nn.Linear(2, 32),
    nn.Tanh(),
    nn.Linear(32, 16),
    nn.Tanh(),
    nn.Linear(16, 8),
    nn.Tanh(),
    nn.Linear(8, 4),
    nn.Tanh(),
    nn.Linear(4, 1)
)

s = Solution(net)
optim = Adam(s.parameters(), lr=1e-3, betas=(0.9, 0.9))
lm = LearningMachine(s)
sampler_pde = ISampler(1000, [[-0.5, 0.5], [-0.5, 0.5]], requires_grad=True)
sampler_b = BoxBoundarySampler(2000, [-0.5, -0.5], [0.5, 0.5], requires_grad=True)


def real(p: torch.Tensor):
    x, y = torch.split(p, 1, dim=-1)
    pi = torch.pi
    return torch.sin(pi*x) * torch.sin(pi*y)

def real_numpy(p):
    return real(torch.from_numpy(p).float()).detach().numpy()


def greal(p: torch.Tensor):
    x, y = torch.split(p, 1, dim=-1)
    pi = torch.pi
    return torch.cat(
        [
        pi*torch.cos(pi*x)*torch.sin(pi*y),
        pi*torch.sin(pi*x)*torch.cos(pi*y),
        ],
        dim=-1
    )


def pde(p: torch.Tensor, u_fn):
    u = u_fn(p)
    u_x, u_y = grad_by_fts(u, p, create_graph=True, split=True)
    u_xx, _ = grad_by_fts(u_x, p, create_graph=True, split=True)
    _, u_yy = grad_by_fts(u_y, p, create_graph=True, split=True)
    f = (2*torch.pi**2 + 1) * torch.sin(torch.pi*p[..., 0:1]) * torch.sin(torch.pi*p[..., 1:2])
    return u - u_xx - u_yy - f


def bc(p: torch.Tensor, u_fn):
    x = p[..., 0]
    y = p[..., 1]
    n = torch.zeros_like(p)
    n[x>torch.abs(y), 0] = 1.0
    n[y>torch.abs(x), 1] = 1.0
    n[x<-torch.abs(y), 0] = -1.0
    n[y<-torch.abs(x), 1] = -1.0

    u = u_fn(p)
    grad_u = grad_by_fts(u, p, create_graph=True, split=False)
    g = (greal(p)*n).sum(dim=-1, keepdim=True) + real(p)
    return (grad_u*n).sum(dim=-1, keepdim=True) + u - g


from fealpy.mesh import MeshFactory as Mf

mesh = Mf.boxmesh2d([-0.5, 0.5, -0.5, 0.5], nx=20, ny=20, meshtype='tri')


for epoch in range(1000):
    optim.zero_grad()
    mse_pde = lm.loss(sampler_pde, pde)
    mse_b = lm.loss(sampler_b, bc)
    loss = mse_pde*0.005 + mse_b*0.995

    loss.backward()
    optim.step()

    if epoch % 100 == 0:
        error = s.estimate_error(real_numpy, mesh, coordtype='c')
        print(f"Epoch: {epoch}, loss: {loss}, error: {error}")


from matplotlib import pyplot as plt
from matplotlib import cm

x = np.linspace(-0.5, 0.5, 40)
y = np.linspace(-0.5, 0.5, 40)

data, (xm, ym) = s.meshgrid_mapping(x, y)
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1, projection='3d')
axes.plot_surface(xm, ym, data, cmap=cm.RdYlBu_r)
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_zlabel('u')
plt.show()
