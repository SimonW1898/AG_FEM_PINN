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

class HelmholtzProblem:
    def __init__(self, nx,ny, T, nt):
        self.nx = nx
        self.ny = ny
        self.mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, dolfinx.mesh.CellType.quadrilateral)
        self.T = T # total time
        self.nt = nt # number of time steps
        self.dt = T / nt # time step size
        self.V = fem.functionspace(self.mesh, ("Lagrange", 1)) # function space
        self.u_exact = self.get_exact_solution() # exact solution
        self.u_analytical = self.get_analytical_solution() # analytical solution
        self.u = ufl.TrialFunction(self.V) # trial function
        self.v = ufl.TestFunction(self.V) # test function
        self.A = self.get_bilinear_form() # bilinear form
        self.L = self.get_linear_form() # linear form
        self.bc = self.get_boundary_condition() # boundary condition
        self.uh = self.solve_problem() # numerical solution
        self.eh = self.uh - self.u_exact # error
        self.eh_analytical = self.uh - self.u_analytical
        self.l2_error = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(self.eh, self.eh) * ufl.dx))).real
        self.l2_error_analytical = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(self.eh_analytical, self.eh_analytical) * ufl.dx))).real
        self.print_results()


    def get_exact_solution(self):
        u_exact = dolfinx.fem.Function(self.V, dtype=np.complex128)
        # watch out here is a factor of 1j/(1j+2 * pi^2*dt) in the exact solution because when computing the error some constant value remains which is 2pi^2*dt
        # u_exact.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-1j * 2 * np.pi**2 * self.dt) * (1j/(1j + 2 * np.pi**2 * self.dt)))
        u_exact.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])  * (1j/(1j + 2 * np.pi**2 * self.dt)))

        return u_exact

    def get_analytical_solution(self):
        u_analytical = dolfinx.fem.Function(self.V, dtype=np.complex128)
        u_analytical.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-1j * 2 * np.pi**2 * self.dt))
        return u_analytical
    
    def get_bilinear_form(self):
        a = (1j / self.dt) * ufl.inner(self.u, self.v) * ufl.dx + ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx
        return a
    
    def get_linear_form(self):
        f = dolfinx.fem.Function(self.V, dtype=np.complex128)
        f.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        L = (1j / self.dt) * ufl.inner(f, self.v) * ufl.dx
        return L
    
    def get_boundary_condition(self):
        self.mesh.topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim)
        boundary_facets = dolfinx.mesh.exterior_facet_indices(self.mesh.topology)
        boundary_dofs = dolfinx.fem.locate_dofs_topological(self.V, self.mesh.topology.dim-1, boundary_facets)
        zero = fem.Constant(self.mesh, PETSc.ScalarType(0.0 + 0.0j))
        bc = fem.dirichletbc(zero, boundary_dofs, self.V)
        return bc
    
    def solve_problem(self):
        problem = LinearProblem(self.A, self.L, bcs=[self.bc])
        uh = problem.solve()
        return uh
    
    def print_results(self):
        print("n = ", self.nx, ", dt = ", self.dt, ", l2 corrected = ", self.l2_error, ", l2 = ", self.l2_error_analytical)


if __name__ == "__main__":
        
    # test the HelmholtzProblem class
    # nx, ny = 16, 16  # number of grid points in x and y direction
    # T = 1.0  # total time
    # nt = 100  # number of time steps
    # helmholtz_problem = HelmholtzProblem(nx, ny, T, nt)

    T = 1.0  # total time
    nt = 10000  # number of time steps
    spatial_resolution = np.array([16, 128, 256, 1024])
    l2_errors = np.zeros_like(spatial_resolution, dtype=np.float64)
    
    for i, n in enumerate(spatial_resolution):
        helmholtz_problem = HelmholtzProblem(n, n, T, nt)
        l2_errors[i] = helmholtz_problem.l2_error

    # plot it
    import matplotlib.pyplot as plt
    plt.loglog(spatial_resolution, l2_errors, marker='o')
    plt.xlabel('n')
    plt.ylabel('L2 error')
    plt.title('L2 error vs. spatial resolution')
    plt.grid()
    plt.show()
    # save it 
    plt.savefig("helmholtz_error.png")


    # # create mesh
    # n = 1000
    # nx, ny = n, n
    # mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, dolfinx.mesh.CellType.triangle)



    # # timestep parameters
    # T = 1.0  # total time
    # nt = 100  # number of time steps
    # dt = T / nt  # time step size


    # # create function space
    # V = fem.functionspace(mesh, ("Lagrange", 1))


    # # exact solution is 
    # u_exact = dolfinx.fem.Function(V, dtype=np.complex128)
    # # watch out here is a factor of 1j/(1j+2 * pi^2*dt) in the exact solution because when computing the error some constant value remains which is 2pi^2*dt
    # # but i'm not getting it seems still wrong
    # u_exact.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-1j * 2 * np.pi**2 * dt)* (1j/(1j+2 * np.pi**2*dt)))


    # ## solve for only one time step

    # # test and trial functions
    # u = ufl.TrialFunction(V)
    # v = ufl.TestFunction(V)
    # # rhs is initial condition f = u0 = sin(pi x_1) sin(pi x_2)
    # f =  dolfinx.fem.Function(V, dtype=np.complex128)
    # f.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    # # bilinear + linear form
    # a = (1j / dt) * ufl.inner(u, v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    # L = (1j / dt) * ufl.inner(f, v) * ufl.dx

    # # define boundary condition
    # mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    # boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    # boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim-1, boundary_facets)
    # zero = fem.Constant(mesh, PETSc.ScalarType(0.0 + 0.0j))
    # bc = fem.dirichletbc(zero, boundary_dofs, V)

    # # solve problem
    # problem = LinearProblem(a, L, bcs=[bc])
    # uh = problem.solve()


    # # compute error
    # error = uh - u_exact

    # # compute L2 norm of error
    # l2_error = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(error, error) * ufl.dx)))
    # # take real part as imaginary should be 0 anyways
    # l2_error = (l2_error.real)
    # print(f"L2 norm of error: {l2_error:.12f}")
