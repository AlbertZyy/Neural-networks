
import torch
import torch.nn as nn

from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh import MeshFactory as Mf
from fealpy.pinn.modules import BoxDBCSolution
from fealpy.pinn.sampler import TriangleMeshSampler
from fealpy.pinn.grad import grad_by_fts


data = CosCosData()

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

box = [0, 1, 0, 1]
s = BoxDBCSolution(pinn)
s.set_box(box)

@s.set_bc
def solution(p: torch.Tensor):
    x, y = p.split(1, dim=-1)
    pi = torch.pi
    val = torch.cos(pi*x) * torch.cos(pi*y)
    return val # val.shape == x.shape


mesh = Mf.boxmesh2d(box=box, nx=2, ny=2)
sampler = TriangleMeshSampler(100, mesh, requires_grad=True)


def pde(p: torch.Tensor, fn):
    u = fn(p)
    u_x, u_y = grad_by_fts(u, p, create_graph=True, split=True)
    u_xx, _ = grad_by_fts(u_x, p, create_graph=True, split=True)
    _, u_yy = grad_by_fts(u_y, p, create_graph=True, split=True)

    return u_xx + u_yy + torch.pi**2 * solution(p)

mse_fn = nn.MSELoss(reduction='mean')
optim = torch.optim.Adam(s.parameters(), lr=0.001, betas=(0.01, 0.1))

iters = 1200

for epoch in range(1, iters+1):
    optim.zero_grad()
    pde_out = pde(sampler.run(), s)
    loss = mse_fn(pde_out, torch.zeros_like(pde_out))
    loss.backward()
    optim.step()

    if epoch % 100 == 0:
        error = s.estimate_error(data.solution, mesh, coordtype='c', squeeze=True, split=True)
        print(f"Epoch: {epoch} | Loss: {loss.data}, Error: {error.sum()}")

    if epoch % 300 == 0:
        measure = mesh.entity_measure('cell')
        refine_flag = error/measure > error.sum()/measure.sum()
        mesh.bisect(isMarkedCell=refine_flag)


from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

x = np.linspace(0, 1, 30)
y = np.linspace(0, 1, 30)
plot_data, (X, Y) = s.meshgrid_mapping(x, y)

fig = plt.figure()
axes = fig.add_subplot(1, 2, 1, projection='3d')
axes.set_zlim([-1, 1])
axes.plot_surface(X, Y, plot_data, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_zlabel('u')

axes = fig.add_subplot(1, 2, 2)
mesh.add_plot(axes)
mesh.find_node(axes, fontsize=12)
plt.show()
