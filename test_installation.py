import dolfinx
print(f"DOLFINx version: {dolfinx.__version__} based on GIT commit: {dolfinx.git_commit_hash} of https://github.com/FEniCS/dolfinx/")

# test for complex numbers
from mpi4py import MPI
import dolfinx
import numpy as np
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u_r = dolfinx.fem.Function(V, dtype=np.float64) 
u_r.interpolate(lambda x: x[0])
u_c = dolfinx.fem.Function(V, dtype=np.complex128)
u_c.interpolate(lambda x:0.5*x[0]**2 + 1j*x[1]**2)
print(u_r.x.array.dtype)
print(u_c.x.array.dtype)

from petsc4py import PETSc 
import numpy as np
from dolfinx.fem.petsc import assemble_vector
print(PETSc.ScalarType)
assert np.dtype(PETSc.ScalarType).kind == 'c'



# check for PyTorch installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


