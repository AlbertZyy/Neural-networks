
import numpy as np


from fealpy.functionspace.Function import Function
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace



T = 1.0
domain = [0, 1, 0, 1]
mesh = MF.boxmesh2d(domain, nx=100, ny=100, meshtype='tri')
space = LagrangeFiniteElementSpace(mesh, p=5)

phi0 = Function(space, array=np.load('LevelSet2d_5.npy'))


from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
phi0.add_plot(ax)

plt.show()
