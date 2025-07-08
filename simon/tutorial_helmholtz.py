import numpy as np

import ufl
from dolfinx import fem
from dolfinx.fem import Function, assemble_scalar, form
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square
from ufl import dx, grad, inner

from mpi4py import MPI
from petsc4py import PETSc

# wavenumber
k0 = 4 * np.pi * 1j

# approximation space polynomial degree
deg = 1

# number of elements in each direction of msh
n_elem = 100

msh = create_unit_square(MPI.COMM_WORLD, n_elem, n_elem)
n = ufl.FacetNormal(msh)

# Source amplitude
if np.issubdtype(PETSc.ScalarType, np.complexfloating):
    A = PETSc.ScalarType(1 + 1j)
else:
    A = 1

# Test and trial function space
V = fem.functionspace(msh, ("Lagrange", deg))

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = Function(V)
f.interpolate(lambda x: A * k0**2 * np.cos(k0 * x[0]) * np.cos(k0 * x[1]))
a = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx
L = inner(f, v) * dx

# Compute solution
uh = Function(V)
uh.name = "u"
problem = LinearProblem(a, L, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
problem.solve()

# Save solution in XDMF format (to be viewed in Paraview, for example)
with XDMFFile(MPI.COMM_WORLD, "out_helmholtz/plane_wave.xdmf", "w", encoding=XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(msh)
    file.write_function(uh)


# Function space for exact solution - need it to be higher than deg
V_exact = fem.functionspace(msh, ("Lagrange", deg + 3))
u_exact = Function(V_exact)
u_exact.interpolate(lambda x: A * np.cos(k0 * x[0]) * np.cos(k0 * x[1]))

# H1 errors
diff = uh - u_exact
H1_diff = msh.comm.allreduce(assemble_scalar(form(inner(grad(diff), grad(diff)) * dx)), op=MPI.SUM)
H1_exact = msh.comm.allreduce(assemble_scalar(form(inner(grad(u_exact), grad(u_exact)) * dx)), op=MPI.SUM)
print("Relative H1 error of FEM solution:", abs(np.sqrt(H1_diff) / np.sqrt(H1_exact)))

# L2 errors
L2_diff = msh.comm.allreduce(assemble_scalar(form(inner(diff, diff) * dx)), op=MPI.SUM)
L2_exact = msh.comm.allreduce(assemble_scalar(form(inner(u_exact, u_exact) * dx)), op=MPI.SUM)
print("Relative L2 error of FEM solution:", abs(np.sqrt(L2_diff) / np.sqrt(L2_exact)))