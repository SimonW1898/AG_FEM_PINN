import numpy as np
import dolfinx
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl
from ufl import conj
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI

import matplotlib.pyplot as plt


class SchroedingerPotentialProblem:
    # Constructor
    def __init__(self, nx, ny, T, nt):
        self.nx = int(nx)
        self.ny = int(ny)
        self.mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, dolfinx.mesh.CellType.triangle)
        self.T = T
        self.nt = int(nt)
        self.dt = T / self.nt
        self.analytical_exists = True  # Flag to indicate if analytical solution exists
        
        # Create function space    
        self.V = fem.functionspace(self.mesh, ("Lagrange", 1))
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)


        # Initial + Boundary Conditions
        self.initial_condition = self.get_initial_condition()
        self.bc = self.get_boundary_condition()
        
        
        # Define functions
        self.u_analytical = self.get_analytical_solution()
        self.uh_old = self.get_initial_condition()
        self.uh_new = dolfinx.fem.Function(self.V, dtype=np.complex128)
        self.potential = self.get_potential()
        

        # Bilinear and Linear Forms
        self.A = self.get_bilinear_form()
        self.L = self.get_linear_form()


        # solve the equation
        self.time_loop()


#methods
    def get_analytical_solution(self, t = 0.0):
        u_analytical = dolfinx.fem.Function(self.V, dtype=np.complex128)
        u_analytical.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-1j * 2 * np.pi**2 * t))
        return u_analytical
    
    def get_initial_condition(self):
        initial_condition = dolfinx.fem.Function(self.V, dtype=np.complex128)
        initial_condition.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        return initial_condition
    
    def get_boundary_condition(self):
        self.mesh.topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)

        boundary_facets = dolfinx.mesh.exterior_facet_indices(self.mesh.topology)
        boundary_dofs = dolfinx.fem.locate_dofs_topological(self.V, self.mesh.topology.dim-1, boundary_facets)

        # Define boundary conditions
        u_D = dolfinx.fem.Function(self.V, dtype=np.complex128)
        u_D.x.array[:] = 0.0
        bc = dolfinx.fem.dirichletbc(u_D, boundary_dofs)
        return bc
    

    def get_potential(self):
        potential = dolfinx.fem.Function(self.V, dtype=np.complex128)
        potential.interpolate(lambda x: 0 * np.exp(-10 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2)))
        return potential
    

    def get_bilinear_form(self):
        """ 
        a(u,v) = (i/dt) ∫_Ω u^{t+dt} * conj(v)
                - ∫_Ω V(x,t) u^{t+dt} * conj(v)
                + ∫_Ω ∇u^{t+dt} · ∇conj(v)
        """ 
        a = (1j / self.dt) * ufl.inner(self.u, self.v) * ufl.dx \
                -  ufl.inner(self.potential * self.u, self.v) * ufl.dx \
                + ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx
        return a
    
    def get_linear_form(self):
        """
        L(u,v) = (i/dt) ∫_Ω u^t * conj(v)
        """
        L = (1j / self.dt) * ufl.inner(self.uh_old, self.v) * ufl.dx
        return L
    
    def time_loop(self):
        problem = LinearProblem(self.A, self.L, bcs=[self.bc])
        for n in range(self.nt):
            # Update time
            t = n * self.dt
            
            # Solve the linear system
            self.uh_new = problem.solve()
            
            # Update old solution
            self.uh_old.x.array[:] = self.uh_new.x.array
            
            
            if self.analytical_exists:
                # Update analytical solution
                self.u_analytical = self.get_analytical_solution(t=t)

                # Calculate error
                self.eh = self.uh_new - self.u_analytical
                # self.eh_array = self.uh_new.x.array - self.u_analytical.x.array
                self.l2_error = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(self.eh, self.eh) * ufl.dx))).real

                print(f"Time step {n+1}/{self.nt}, L2 Error: {self.l2_error:.6f}")



if __name__ == "__main__":
    # Example usage
    nx = 10  # Number of elements in x-direction
    ny = 10  # Number of elements in y-direction
    T = 1.0  # Total time
    nt = 100  # Number of time steps

    problem = SchroedingerPotentialProblem(nx, ny, T, nt)
    
    # Visualization (optional)

