# solve the simplest case of the helmholtz equation
# $$i dt u = - laplacian u$$
# $$ u(x,t) = 0, \forall x \in \partial \Omega, t>0$$
# $$ u(x,0) = 1, \forall x \in \Omega$$
# Exact solution is $ u(x,t) = \sin(\pi x_1)\sin(\pi x_2) e^{-i 2\pi^2 t}$

# imports
import numpy as np
import dolfinx
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl
from ufl import conj
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI


# create mesh
n = 1000
nx, ny = n, n
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, dolfinx.mesh.CellType.triangle)



# timestep parameters
T = 1.0  # total time
nt = 100  # number of time steps
dt = T / nt  # time step size


# create function space
V = fem.functionspace(mesh, ("Lagrange", 1))


# exact solution is 
u_exact = dolfinx.fem.Function(V, dtype=np.complex128)
# watch out here is a factor of 1j/(1j+2 * pi^2*dt) in the exact solution because when computing the error some constant value remains which is 2pi^2*dt
# but i'm not getting it seems still wrong
u_exact.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-1j * 2 * np.pi**2 * dt)* (1j/(1j+2 * np.pi**2*dt)))


## solve for only one time step

# test and trial functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
# rhs is initial condition f = u0 = sin(pi x_1) sin(pi x_2)
f =  dolfinx.fem.Function(V, dtype=np.complex128)
f.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

# bilinear + linear form
a = (1j / dt) * ufl.inner(u, v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (1j / dt) * ufl.inner(f, v) * ufl.dx

# define boundary condition
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim-1, boundary_facets)
zero = fem.Constant(mesh, PETSc.ScalarType(0.0 + 0.0j))
bc = fem.dirichletbc(zero, boundary_dofs, V)

# solve problem
problem = LinearProblem(a, L, bcs=[bc])
uh = problem.solve()


# compute error
error = uh - u_exact

# compute L2 norm of error
l2_error = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(error, error) * ufl.dx)))
# take real part as imaginary should be 0 anyways
l2_error = (l2_error.real)
print(f"L2 norm of error: {l2_error:.12f}")
