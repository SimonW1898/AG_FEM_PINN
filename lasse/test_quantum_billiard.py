#!/usr/bin/env python3
"""
Minimal test implementation of the Quantum Billiard solution from the notebook.
This is the exact same code as in the working QuantumBilliard notebook.
"""

import numpy as np
import dolfinx
from dolfinx import mesh, fem, io
import ufl
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI

def test_quantum_billiard():
    """Test the quantum billiard solution exactly as in the notebook."""
    
    # 1. Mesh and function space
    n = 16
    nx, ny = n, n
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))

    # Boundary condition (Dirichlet homogeneous)
    uD = fem.Function(V)
    uD.interpolate(lambda x: np.zeros(x.shape[1]))

    # Create facet to cell connectivity required to determine boundary facets
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)

    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)

    # 3. Weak form: A u = E M u
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx  # Stiffness
    m = ufl.inner(u, v) * ufl.dx                      # Mass

    from dolfinx.fem.petsc import assemble_matrix
    # assemble
    A = assemble_matrix(fem.form(a), bcs=[bc])
    A.assemble()
    M = assemble_matrix(fem.form(m), bcs=[bc])
    M.assemble()

    # Solve the Eigenvalue problem
    eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
    eigensolver.setOperators(A, M)
    eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)  # Generalized Hermitian
    eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    eigensolver.setFromOptions()
    eigensolver.solve()

    n_conv = eigensolver.getConverged()

    print(f"Number of converged eigenpairs: {n_conv}")
    print("First few eigenvalues:")
    
    for i in range(min(n_conv, 5)):
        eig_val = eigensolver.getEigenvalue(i)
        x = V.tabulate_dof_coordinates()

        # Allocate vector for eigenfunction
        r, _ = A.getVecs()
        eigensolver.getEigenvector(i, r)

        # Create Function in V and assign vector
        u_eig = fem.Function(V)
        u_eig.x.array[:] = r.getArray()

        if MPI.COMM_WORLD.rank == 0:
            print(f"Eigenvalue {i}: {eig_val}")

    # Get the first eigenfunction for further analysis
    r, _ = A.getVecs()
    eigensolver.getEigenvector(0, r)
    u_eig = fem.Function(V)
    u_eig.x.array[:] = r.getArray()

    # Print some statistics about the first eigenfunction
    if MPI.COMM_WORLD.rank == 0:
        print(f"\nFirst eigenfunction statistics:")
        print(f"Max value: {np.max(np.abs(u_eig.x.array)):.6f}")
        print(f"Min value: {np.min(np.abs(u_eig.x.array)):.6f}")
        print(f"Mean value: {np.mean(np.abs(u_eig.x.array)):.6f}")
        
        # Check boundary values
        boundary_mask = ((x[:, 0] < 0.01) | (x[:, 0] > 0.99) | 
                        (x[:, 1] < 0.01) | (x[:, 1] > 0.99))
        boundary_values = u_eig.x.array[boundary_mask]
        max_boundary = np.max(np.abs(boundary_values))
        print(f"Max boundary value: {max_boundary:.6f}")

    print("Quantum billiard test completed successfully!")

if __name__ == "__main__":
    test_quantum_billiard() 