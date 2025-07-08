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

import matplotlib.pyplot as plt

class HelmholtzProblem:
    def __init__(self, nx,ny, T, nt):
        self.nx = int(nx)
        self.ny = int(ny)
        self.mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, dolfinx.mesh.CellType.triangle)
        self.T = T
        self.nt = int(nt)
        self.dt = T / self.nt
        self.V = fem.functionspace(self.mesh, ("Lagrange", 1))
        self.u_exact = self.get_exact_solution()
        self.u_analytical = self.get_analytical_solution()
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        self.initial_condition = self.get_initial_condition()
        self.A = self.get_bilinear_form()
        self.L = self.get_linear_form()
        self.bc = self.get_boundary_condition()
        self.uh = self.solve_problem()
        self.eh = self.uh - self.u_exact
        self.eh_array = self.uh.x.array - self.u_analytical.x.array
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
    
    def get_initial_condition(self):
        f = dolfinx.fem.Function(self.V, dtype=np.complex128)
        f.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        return f
    
    def get_linear_form(self):
        L = (1j / self.dt) * ufl.inner(self.initial_condition, self.v) * ufl.dx
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
        # print the results
        print(f"n = {self.nx}, dt = {self.dt:.2e}, L2 error (corrected) = {self.l2_error:.6e}, L2 error (analytical) = {self.l2_error_analytical:.6e}")

    def plot_solution(self):
        import pyvista
        from dolfinx import plot
   
        pyvista.start_xvfb()
        topology, cell_types, geometry = plot.vtk_mesh(self.mesh)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

        # Create a VTK-compatible mesh
        topology, cell_types, geometry = plot.vtk_mesh(self.mesh, self.mesh.topology.dim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

        # Add the solution as a point data array
        grid.point_data["u_real"] = self.uh.x.array.real
        grid.point_data["u_imag"] = self.uh.x.array.imag
        grid.point_data["e_h"] = self.eh_array.imag  # Use real part of the error for visualization

        # Plot and save (off-screen)
        pyvista.OFF_SCREEN = True
        plotter = pyvista.Plotter(off_screen=True)
        plotter.add_mesh(grid, scalars="e_h", cmap="viridis")
        plotter.view_xy()
        plotter.show()
        plotter.screenshot("helmholtz_solution.png")
        print("Plot saved as helmholtz_solution.png")




def plot_convergence(spatial_resolution, l2_errors):
    import matplotlib.pyplot as plt
    plt.loglog(spatial_resolution, l2_errors, marker='o')
    plt.xlabel('Spatial Resolution (n)')
    plt.ylabel('L2 Error')
    plt.title('L2 Error vs. Spatial Resolution')
    plt.grid()
    plt.show()
    # save it 
    plt.savefig("helmholtz_error.png")
    print("Plot saved as helmholtz_error.png")

    

if __name__ == "__main__":
        
    # test the HelmholtzProblem class
    # nx, ny = 16, 16  # number of grid points in x and y direction
    # T = 1.0  # total time
    # nt = 100  # number of time steps
    # helmholtz_problem = HelmholtzProblem(nx, ny, T, nt)

    T = 0.1  # total time
    nt = 1e4  # number of time steps
    spatial_resolution = np.array([16, 128])#, 256, 1024])
    l2_errors = np.zeros_like(spatial_resolution, dtype=np.float64)
    
    for i, n in enumerate(spatial_resolution):
        helmholtz_problem = HelmholtzProblem(n, n, T, nt)
        l2_errors[i] = helmholtz_problem.l2_error


    plot_convergence(spatial_resolution, l2_errors)
    helmholtz_problem.plot_solution()

    



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
