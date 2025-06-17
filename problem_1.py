# solve the simplest case of the helmholtz equation
# i dt u = - laplacian u

# imports
import numpy as np
import dolfinx
from dolfinx import mesh, fem, io
# import ufl
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI


from dolfinx.fem.petsc import assemble_vector
print(PETSc.ScalarType)
assert np.dtype(PETSc.ScalarType).kind == 'c'

