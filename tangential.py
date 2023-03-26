
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from fealpy.pinn.machine import Solution
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
    nn.Linear(8, 3)
)


def c(p: torch.Tensor) ->torch.Tensor:
    mask = p[...,0] <=0
    shape = p.shape
    result = torch.zeros(shape,dtype=torch.float32 )
    result[mask, :] = torch.sin(np.pi*mkfs(p[mask, :][..., 1:2], p[mask, :][..., 0:1]))
    result[~mask, :] = mkfs(2*torch.sin(np.pi*p[~mask, :][..., 1:2]), torch.sin(np.pi*p[~mask, :][..., 0:1]))

    return result


class Ext(Solution):
    def forward(self, p: Tensor):
        mask = p[..., 0] < 0
        output = self.net(p)
        ret = torch.zeros(p.shape[:-1] + (2,), dtype=torch.float32)
        idx_l = torch.tensor([True, False, True])
        idx_r = torch.tensor([False, True, True])
        ret[mask, :] = output[mask, :][..., idx_l]
        ret[~mask, :] = output[~mask, :][..., idx_r]
        return ret


class Hat(Solution):
    def forward(self, p: Tensor):
        shape = tuple(p.shape[:-1]) + (1, )
        y_1 = mkfs(-1, p[..., 1:2])
        y_2 = mkfs(1, p[..., 1:2])
        x_1 = mkfs(p[..., 0:1], -1)
        x_2 = mkfs(p[..., 0:1], 1)
        c_1 = mkfs(-1, -1.0, f_shape=shape)
        c_2 = mkfs(1.0, -1.0, f_shape=shape)
        c_3 = mkfs(-1, 1.0, f_shape=shape)
        c_4 = mkfs(1.0, 1.0, f_shape=shape)

        return self.net(p) + 0.5*(1-p[...,0:1])*(c(y_1)-self.net(y_1)) + 0.5*(1+p[...,0:1])*(c(y_2)-self.net(y_2))\
        + 0.5*(1-p[...,1:2])*(c(x_1)-self.net(x_1) + 0.5*(1-p[...,0:1])*(self.net(c_1)-c(c_1)) + 0.5*(1+p[...,0:1])*(self.net(c_2)-c(c_2)))\
        + 0.5*(1+p[...,1:2])*(c(x_2)-self.net(x_2) + 0.5*(1-p[...,0:1])*(self.net(c_3)-c(c_3)) + 0.5*(1+p[...,0:1])*(self.net(c_4)-c(c_4)))


netT = Hat(Ext(net))


def pde(p: Tensor):

    u = netT(p)
    u1 = u[..., 0:1]
    u2 = u[..., 1:2]

    mask = p[..., 0] <= 0

    _, u1_y = gradient(u1, p, create_graph=True, split=True)
    u2_x, _ = gradient(u2, p, create_graph=True, split=True)

    _, u1_yy = gradient(u1_y, p, create_graph=True, split=True)
    _, u2_xy = gradient(u2_x, p, create_graph=True, split=True)
    r1 = torch.zeros((p.shape[0], 1), dtype=torch.float32)
    r1[mask, :] = u2_xy[mask, :] - u1_yy[mask, :] - u1[mask, :] - (np.pi**2 - 1)*torch.sin(np.pi*p[mask, 1:2])
    r1[~mask, :] = u2_xy[~mask, :] - u1_yy[~mask, :] - u1[~mask, :] - (2*np.pi**2 - 1)*torch.sin(np.pi*p[~mask, 1:2])

    u2_xx, _ = gradient(u2_x, p, create_graph=True, split=True)
    u1_yx, _ = gradient(u1_y, p, create_graph=True, split=True)
    r2 = u1_yx - u2_xx - u2 - (np.pi**2 - 1)*torch.sin(np.pi*p[..., 0:1])

    return torch.cat([r1, r2], dim=1)


def real_1(p: np.ndarray):
    yx = np.concatenate([p[..., 1:2], p[..., 0:1]], axis=-1)
    return np.sin(np.pi*yx)

def real_2(p: np.ndarray):
    return np.concatenate([2*np.sin(np.pi*p[..., 1:2]), np.sin(np.pi*p[..., 0:1])], axis = -1)

def realTotal(p: np.ndarray):
    mask = p[..., 0] <=0
    shape = p.shape
    result = np.zeros(shape, dtype=np.float32)
    result[mask,:] = real_1(p[mask, :])
    result[~mask,:] = real_2(p[~mask, :])

    return result


sampler1 = ISampler(1000, [[-1, 1], [-1, 1]], requires_grad=True)
optim = Adam(net.parameters(), lr=1e-3, betas=(0.01, 0.1))
mse_cost_fn = nn.MSELoss(reduction='mean')

mesh = Mf.boxmesh2d([-1, 1, -1, 1], nx=20, ny=20)

errors = []

for epoch in range(500):
    optim.zero_grad()
    output = pde(sampler1.run())
    loss = mse_cost_fn(output, torch.zeros_like(output))
    loss.backward()

    error = netT.estimate_error(realTotal, mesh=mesh, coordtype='c')
    errors.append(error)
    optim.step()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}, Error:{error}")


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
