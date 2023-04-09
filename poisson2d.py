import numpy as np
# solver
from scipy.sparse.linalg import spsolve

from matplotlib import pyplot as plt
from matplotlib import rc
# rc('text', usetex=True)

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table


degree = 2
GD = 2
nrefine = 2
maxit = 1

if GD == 2:
    from fealpy.pde.poisson_2d import CosCosData as PDE
elif GD == 3:
    from fealpy.pde.poisson_3d import CosCosCosData as PDE

pde = PDE()
mesh = pde.init_mesh(n=nrefine)

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

for i in range(maxit):
    print("The {}-th computation:".format(i))
    space = LagrangeFiniteElementSpace(mesh, p=degree)
    NDof[i] = space.number_of_global_dofs()
    bc = DirichletBC(space, pde.dirichlet)

    uh = space.function() # uh 即是一个有限元函数，也是一个数组
    A = space.stiff_matrix() # (\nabla uh, \nabla v_h)
    F = space.source_vector(pde.source) # (f, vh)
    A, F = bc.apply(A, F, uh)
    uh[:] = spsolve(A, F)

    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value)

    if i < maxit-1:
        mesh.uniform_refine()

if GD == 2:
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1, projection='3d')
    uh.add_plot(axes, cmap='rainbow')
elif GD == 3:
    print('The 3d function plot is not been implemented!')


###

from fealpy.pinn.modules import BoxDBCSolution
import torch.nn as nn
from torch import Tensor
import torch


pinn = nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 8),
    nn.Tanh(),
    nn.Linear(8, 4),
    nn.Tanh(),
    nn.Linear(4, 1)
)

s = BoxDBCSolution(pinn)
s.set_box([0, 1, 0, 1])
@s.set_bc
def bc(p: Tensor):
    return torch.zeros_like(p, requires_grad=True)

for epoch in range(1, 101):
    ...

plt.show()
