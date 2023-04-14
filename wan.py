
import numpy as np
import torch
import torch.nn as nn

from fealpy.pinn.modules import BoxDBCSolution
from fealpy.pinn.sampler import ISampler, TriangleMeshSampler
from fealpy.pinn.grad import grad_by_fts

from fealpy.mesh import MeshFactory as Mf

pinn = nn.Sequential(
    nn.Linear(2, 64),
    nn.Tanh(),
    nn.Linear(64, 32),
    nn.Tanh(),
    nn.Linear(32, 16),
    nn.Tanh(),
    nn.Linear(16, 8),
    nn.Tanh(),
    nn.Linear(8, 4),
    nn.Tanh(),
    nn.Linear(4, 1)
)

box = [0, 2, 0, 2]

s = BoxDBCSolution(pinn)
s.set_box(box)
@s.set_bc
def bc(p: torch.Tensor):
    x, y = p.split(1, dim=-1)
    pi = torch.pi
    return torch.cos(pi*x)*torch.cos(pi*y)


pinn2 = nn.Sequential(
    nn.Linear(2, 64),
    nn.Tanh(),
    nn.Linear(64, 32),
    nn.Tanh(),
    nn.Linear(32, 16),
    nn.Tanh(),
    nn.Linear(16, 8),
    nn.Tanh(),
    nn.Linear(8, 4),
    nn.Tanh(),
    nn.Linear(4, 1)
)

t = BoxDBCSolution(pinn2)
t.set_box(box)
@t.set_bc
def bc2(p: torch.Tensor):
    return torch.zeros(p.shape[:-1]+(1, ), device=p.device)

mesh = Mf.boxmesh2d(box, nx=25, ny=25)

# sampler = ISampler(10000, [[0, 2], [0, 2]], requires_grad=True)
sampler = TriangleMeshSampler(10, mesh, requires_grad=True)
optim_u = torch.optim.Adadelta(s.parameters(), lr=1e-0, rho=0.9)
optim_p = torch.optim.Adadelta(t.parameters(), lr=1e-0, rho=0.9, maximize=True)

def loss(p: torch.Tensor, fn, test):
    u = fn(p)
    phi = test(p)

    grad_u = grad_by_fts(u, p, create_graph=True)
    grad_phi = grad_by_fts(phi, p, create_graph=True)

    mul = torch.sum(grad_u*grad_phi, dim=-1, keepdim=True) - 2*torch.pi**2*phi*bc(p)
    inner = torch.mean(mul, dim=0)
    norm_phi = torch.mean(phi**2, dim=0)
    return torch.log(inner**2) - torch.log(norm_phi)


def real(p: np.ndarray):
    x = p[..., 0:1]
    y = p[..., 1:2]
    pi = np.pi
    return np.cos(pi*x)*np.cos(pi*y)


for epoch in range(1, 901):
    inputs = sampler.run()
    loss_data = loss(inputs, s, t)

    optim_u.zero_grad()
    optim_p.zero_grad()

    loss_data.backward()

    if (epoch - 1) // 100 in {0, 1, 2, 4, 5, 6, 8, 9}:
        optim_u.step()
    else:
        optim_p.step()

    if epoch % 100 == 0:
        error = s.estimate_error(real, mesh, coordtype='c')
        print(f"Epoch: {epoch}, loss: {loss_data.data}, error: {error}")


from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

x = np.linspace(0, 2, 15)
y = np.linspace(0, 2, 15)

plot_data, (X, Y) = s.meshgrid_mapping(x, y)
plot_data_2, _ = t.meshgrid_mapping(x, y)

fig = plt.figure()
axes = fig.add_subplot(1, 2, 1, projection='3d')
axes.set_zlim([-1, 1])
axes.plot_surface(X, Y, plot_data, cmap=cm.RdYlBu_r)

axes = fig.add_subplot(1, 2, 2, projection='3d')
# axes.set_zlim([-1, 1])
axes.plot_surface(X, Y, plot_data_2, cmap=cm.RdYlBu_r)
plt.show()
