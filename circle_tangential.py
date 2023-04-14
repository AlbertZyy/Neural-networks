import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from fealpy.pinn.modules import TensorMapping, Solution, BoxDBCSolution
from fealpy.pinn.tools import mkfs
from fealpy.pinn.grad import gradient
from fealpy.pinn.sampler import ISampler
from fealpy.mesh import MeshFactory as Mf


net = nn.Sequential(
    nn.Linear(2, 32),
    nn.Tanh(),
    nn.Linear(32, 16),
    nn.Tanh(),
    nn.Linear(16, 8),
    nn.Tanh(),
    nn.Linear(8, 4),
    nn.Tanh(),
    nn.Linear(4, 2)
)

net_2 = nn.Sequential(
    nn.Linear(2, 32),
    nn.Tanh(),
    nn.Linear(32, 16),
    nn.Tanh(),
    nn.Linear(16, 8),
    nn.Tanh(),
    nn.Linear(8, 2)
)


outer_net = BoxDBCSolution(net)
outer_net.set_box([-1, 1, -1, 1])
@outer_net.set_bc
def c(p: torch.Tensor) -> torch.Tensor:
    pi = torch.pi
    x, y = p.split(1, dim=-1)
    result = mkfs(2*torch.sin(pi*y), \
                    2*(x**2 + y**2 - 0.25)\
                    * torch.sin(pi*x))
    return result


class CircleDBCSolution(Solution):
    def forward(self, p: Tensor):
        r = 0.5
        u = self.net(p)
        x, y = p.split(1, dim=-1)
        tangent = mkfs(-y, x)/r
        return u - torch.sum(u*tangent, dim=-1, keepdim=True)*tangent


inner_net = CircleDBCSolution(net_2)


class FinalModule(TensorMapping):
    def __init__(self, inner: nn.Module, outer: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.outer = outer

    def forward(self, p: Tensor):
        mask = p[..., 0]**2 + p[..., 1]**2 < 0.25
        ret = self.outer(p)
        ret[mask, :] += self.inner(p[mask, :])
        return ret


netT = FinalModule(inner=inner_net, outer=outer_net)


def pde(p: Tensor):

    u = netT(p)
    pi = torch.pi
    u1 = u[..., 0:1]
    u2 = u[..., 1:2]

    mask = p[..., 0]**2 + p[..., 1]**2 < 0.25

    _, u1_y = gradient(u1, p, create_graph=True, split=True)
    u2_x, _ = gradient(u2, p, create_graph=True, split=True)

    u1_yx, u1_yy = gradient(u1_y, p, create_graph=True, split=True)
    u2_xx, u2_xy = gradient(u2_x, p, create_graph=True, split=True)


    r1 = torch.zeros((p.shape[0], 1), dtype=torch.float32)

    r1[mask, :] = u2_xy[mask, :] - u1_yy[mask, :] - u1[mask, :]\
                + (pi**2 + 1)*torch.sin(pi*p[mask, 1:2])\
                - 2*pi*p[mask, 1:2]*torch.cos(pi*p[mask, 0:1])

    r1[~mask, :] = u2_xy[~mask, :] - u1_yy[~mask, :] - u1[~mask, :]\
                + 2*(pi**2 + 1)*torch.sin(pi*p[~mask, 1:2])\
                - 4*pi*p[~mask, 1:2]*torch.cos(pi*p[~mask, 0:1])

    r2 = torch.zeros((p.shape[0], 1), dtype=torch.float32)

    r2[mask, :] = u1_yx[mask, :] - u2_xx[mask, :] - u2[mask, :] \
                - (pi**2 - 1)*(p[mask,0:1]**2 + p[mask,1:2]**2 - 0.25) * torch.sin(pi*p[mask, 0:1])\
                + 4*pi*p[mask,0:1]*torch.cos(pi*p[mask,0:1])\
                + 2*torch.sin(pi*p[mask,0:1])

    r2[~mask, :] = u1_yx[~mask, :] - u2_xx[~mask, :] - u2[~mask, :] \
                - 2*(pi**2 - 1)*(p[~mask,0:1]**2 + p[~mask,1:2]**2 - 0.25) * torch.sin(pi*p[~mask, 0:1])\
                + 8*pi*p[~mask,0:1]*torch.cos(pi*p[~mask,0:1])\
                + 4*torch.sin(pi*p[~mask,0:1])

    return torch.cat([r1, r2], dim=-1)


def real_1(p: np.ndarray):

    pi = np.pi
    x = p[...,0:1]
    y = p[...,1:2]
    u_1 = np.sin(pi*y)
    u_2 = (x**2+y**2-0.25)*np.sin(pi*x)

    return np.concatenate([u_1, u_2], axis=-1)

def real_2(p: np.ndarray):

    pi = np.pi
    x = p[...,0:1]
    y = p[...,1:2]
    u_1 = 2*np.sin(pi*y)
    u_2 = 2*(x**2+y**2-0.25)*np.sin(pi*x)

    return np.concatenate([u_1, u_2], axis=-1)

def realTotal(p: np.ndarray):

    mask = p[...,0]**2 + p[...,1]**2 <= 0.25
    shape = p.shape
    result = np.zeros(shape, dtype=np.float32)
    result[mask,:] = real_1(p[mask, :])
    result[~mask,:] = real_2(p[~mask, :])

    return result


sampler1 = ISampler(100, [[-1, 1], [-1, 1]], requires_grad=True)
optim = Adam(net.parameters(), lr = 0.001, betas=[0.9,0.99])
mse_cost_fn = nn.MSELoss(reduction='mean')

mesh = Mf.boxmesh2d([-1, 1, -1, 1], nx=20, ny=20)

errors = []

for epoch in range(200):
    optim.zero_grad()
    output = pde(sampler1.run())
    loss = mse_cost_fn(output, torch.zeros_like(output))
    loss.backward()

    error = netT.estimate_error(realTotal, mesh=mesh, coordtype='c')
    errors.append(error)
    optim.step()

    if epoch % 20 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}, Error:{error}.")


from matplotlib import pyplot as plt

x = np.linspace(-1, 1, 30, dtype=np.float32)
y = np.linspace(-1, 1, 30, dtype=np.float32)

ret, (mx, my) = netT.meshgrid_mapping(x, y)

fig, ax = plt.subplots(figsize=(6, 6))
ax.quiver(mx, my, ret[0], ret[1])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])

fig = plt.subplots(figsize=(6, 6))
y= range(1, len(errors) + 1)

plt.plot(y, errors)
plt.show()
