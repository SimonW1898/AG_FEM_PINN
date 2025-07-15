"""
Modularized Schrödinger Equation Solver using Finite Elements

This module provides a class-based interface for solving the time-dependent 
Schrödinger equation using the finite element method with DOLFINx.

The solver implements the backward Euler time discretization scheme for the 
time-dependent Schrödinger equation:
    i ∂u/∂t = -Δu + V(x,t)u

with homogeneous Dirichlet boundary conditions on the unit square [0,1]².
"""

import numpy as np
import tqdm
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl
from petsc4py import PETSc
from mpi4py import MPI
from slepc4py import SLEPc
import matplotlib.pyplot as plt
from typing import Callable, Optional, Union, Tuple, List, Any
from abc import ABC, abstractmethod
from potentials import ModelPotential

class Potential:
    def __init__(self, func: Callable[[np.ndarray, float], np.ndarray]):
        self.func = ModelPotential(
            time_dependent=True,
            laser_amplitude=0.4,
            laser_omega=5.0,
            laser_pulse_duration=0.4,
            laser_center_time=0.5,
            laser_envelope_type='gaussian',
            laser_spatial_profile_type='uniform',
            laser_charge=1.0,
            laser_polarization='linear_xy'
        )
        self.t = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.func(x, self.t)


class StationarySchrodingerSolver:
    """
    Solves the time-independent Schrödinger equation:
        -Δφ + V(x, y) φ = E φ
    on the unit square with Dirichlet BCs.
    """

    def __init__(self,
                 potential: Callable[[np.ndarray], np.ndarray],
                 nx: int = 64,
                 ny: int = 64):
        self.nx = nx
        self.ny = ny
        self.mesh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
        self.V_space = fem.functionspace(self.mesh, ("Lagrange", 1))

        self.V_func = fem.Function(self.V_space)
        self.V_func.interpolate(potential)

        self.A, self.M = self._assemble_operators()
        self.ground_state = None  # to be computed later

    def _assemble_operators(self):
        u = ufl.TrialFunction(self.V_space)
        v = ufl.TestFunction(self.V_space)

        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(self.V_func * u, v) * ufl.dx
        m = ufl.inner(u, v) * ufl.dx

        A = fem.petsc.assemble_matrix(fem.form(a), bcs=self._get_bcs())
        A.assemble()
        M = fem.petsc.assemble_matrix(fem.form(m), bcs=self._get_bcs())
        M.assemble()

        return A, M

    def _get_bcs(self):
        self.mesh.topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim)
        facets = dolfinx.mesh.exterior_facet_indices(self.mesh.topology)
        dofs = fem.locate_dofs_topological(self.V_space, self.mesh.topology.dim - 1, facets)
        zero = fem.Constant(self.mesh, PETSc.ScalarType(0.0))
        return [fem.dirichletbc(zero, dofs, self.V_space)]

    def solve_ground_state(self):
        """Solve for the lowest eigenvalue and eigenfunction."""
        eps = SLEPc.EPS().create()
        eps.setOperators(self.A, self.M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        # eps.setDimensions(1, PETSc.DECIDE)
        eps.setFromOptions()
        eps.solve()

        if eps.getConverged() < 1:
            raise RuntimeError("No eigenpairs found.")

        vr, _ = self.A.getVecs()
        eps.getEigenpair(0, vr, None)

        phi = fem.Function(self.V_space, dtype=PETSc.ScalarType)
        phi.x.array[:] = vr.array[:]

        norm_sq = fem.assemble_scalar(fem.form(ufl.inner(phi, phi) * ufl.dx))
        phi.x.array[:] /= np.sqrt(norm_sq)

        self.ground_state = phi
        return phi

    def plot_2d(self, plot_type='abs', title='Ground state', save_path=None, show=True):
        if self.ground_state is None:
            raise RuntimeError("Call solve_ground_state() first.")

        X = self.V_space.tabulate_dof_coordinates()
        if plot_type == 'real':
            Z = self.ground_state.x.array.real
            label = 'Re(ϕ)'
        elif plot_type == 'imag':
            Z = self.ground_state.x.array.imag
            label = 'Im(ϕ)'
        else:
            Z = np.abs(self.ground_state.x.array)**2
            label = '|ϕ|²'

        plt.figure(figsize=(6, 5))
        sc = plt.scatter(X[:, 0], X[:, 1], c=Z, cmap='viridis', s=10)
        plt.colorbar(sc, label=label)
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()

    def plot_3d(self, plot_type='abs', title='Ground state (3D)', save_path=None, show=True):
        if self.ground_state is None:
            raise RuntimeError("Call solve_ground_state() first.")

        X = self.V_space.tabulate_dof_coordinates()
        if plot_type == 'real':
            Z = self.ground_state.x.array.real
            label = 'Re(ϕ)'
        elif plot_type == 'imag':
            Z = self.ground_state.x.array.imag
            label = 'Im(ϕ)'
        else:
            Z = np.abs(self.ground_state.x.array)**2
            label = '|ϕ|²'

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(X[:, 0], X[:, 1], Z, cmap='viridis', linewidth=0)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(label)
        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()


class SchrodingerSolver:
    """
    Finite Element solver for the time-dependent Schrödinger equation.
    
    Solves: i ∂u/∂t = -Δu + V(x,t)u
    with homogeneous Dirichlet boundary conditions on [0,1]².
    
    Uses backward Euler time discretization and P1 finite elements.
    """
    
    def __init__(self, 
                 nx: int = 64,
                 ny: int = 64,
                 T_final: float = 1.0,
                 N_time: int = 100,
                 potential: Optional[Any] = None,
                 initial_condition: Optional[Callable] = None,
                 analytical_solution: Optional[Callable] = None):
        """
        Initialize the Schrödinger equation solver.
        
        Parameters:
        - nx, ny: Number of grid points in x and y directions
        - T_final: Final time for simulation
        - N_time: Number of time steps
        - potential: Potential function (Potential class instance)
        - initial_condition: Initial condition function u₀(x,y)
        - analytical_solution: Analytical solution function u(x,y,t) for error analysis
        """
        # Grid and time parameters
        self.nx = int(nx)
        self.ny = int(ny)
        self.T_final = T_final
        self.N_time = int(N_time)
        self.dt = T_final / N_time
        
        # Create mesh and function space
        self.mesh = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, self.nx, self.ny, dolfinx.mesh.CellType.triangle
        )
        self.V = fem.functionspace(self.mesh, ("Lagrange", 1))
        
        # Set potential
        if potential is not None:
            self.potential = Potential(potential)
        else:
            self.potential = None
        self.potential_function = fem.Function(self.V)
            
        # Set initial condition (default to sin(πx)sin(πy))
        if initial_condition is None:
            self.initial_condition = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        else:
            self.initial_condition = initial_condition
            
        # Set analytical solution (default to free particle solution)
        if analytical_solution is None:
            self.analytical_solution = lambda x, t: (
                np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * 
                np.exp(-1j * 2 * np.pi**2 * t)
            )
        else:
            self.analytical_solution = analytical_solution
        
        # Initialize solution storage
        self.solutions: List[Any] = []  # Store solutions at each time step
        self.times = np.linspace(0, T_final, N_time + 1)
        self.errors: List[float] = []  # Store L2 errors if analytical solution is available
        
        # Set up boundary conditions
        self._setup_boundary_conditions()
        
        # Initialize current solution
        self.u_current = None
        self.u_previous = None
        
        print(f"Schrödinger solver initialized:")
        print(f"  Grid: {nx}×{ny}, Time steps: {N_time}, dt = {self.dt:.6f}")
        if self.potential:
            print(f"  Potential: {getattr(self.potential, 'name', 'Custom')}")
        else:
            print(f"  Potential: Zero (free particle)")
        print(f"  Domain: [0,1]²")
    
    def _setup_boundary_conditions(self):
        """Set up homogeneous Dirichlet boundary conditions."""
        # Create connectivity for boundary detection
        self.mesh.topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim)
        
        # Find boundary facets and DOFs
        boundary_facets = dolfinx.mesh.exterior_facet_indices(self.mesh.topology)
        boundary_dofs = dolfinx.fem.locate_dofs_topological(
            self.V, self.mesh.topology.dim - 1, boundary_facets
        )
        
        # Create zero boundary condition
        zero = fem.Constant(self.mesh, np.complex128(0.0 + 0.0j))
        self.bc = fem.dirichletbc(zero, boundary_dofs, self.V)
    
    def set_initial_condition(self, u0_func: Callable):
        """Set the initial condition function."""
        self.initial_condition = u0_func
    
    def set_potential(self, potential: Any):
        """Set the potential function."""
        self.potential = potential
    
    def set_analytical_solution(self, analytical_func: Callable):
        """Set the analytical solution for error analysis."""
        self.analytical_solution = analytical_func
    
    def _create_initial_solution(self):
        """Create the initial solution from the initial condition."""
        u_init = fem.Function(self.V, dtype=np.complex128)
        u_init.interpolate(self.initial_condition)
        return u_init
    
    def _create_variational_forms(self, t_current: float):
        """Create the variational forms for the time step."""
        # Trial and test functions
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        # Update potential function
        if self.potential is not None:
            self.potential.t = t_current
            self.potential_function.interpolate(self.potential)
        
        # Constants for complex arithmetic
        i_dt = fem.Constant(self.mesh, np.complex128(1j / self.dt))
        
        # Bilinear form: (i/dt)⟨u,v⟩ + ⟨∇u,∇v⟩
        if self.potential is not None:
            a = i_dt * ufl.inner(u, v) * ufl.dx - ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - ufl.inner(self.potential_function * u, v) * ufl.dx
        else:
            a = i_dt * ufl.inner(u, v) * ufl.dx - ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        
        # Linear form: (i/dt)⟨u_prev,v⟩
        if self.potential is not None:
            L = i_dt * ufl.inner(self.u_previous, v) * ufl.dx
        else:
            L = i_dt * ufl.inner(self.u_previous, v) * ufl.dx
        
        return a, L
    
    def _solve_time_step(self, t_current: float):
        """Solve a single time step using backward Euler."""
        # Create variational forms
        a, L = self._create_variational_forms(t_current)
        
        # Solve linear system
        problem = LinearProblem(a, L, bcs=[self.bc])
        u_new = problem.solve()
        
        return u_new
    
    def solve(self, store_solutions: bool = True, compute_errors: bool = True):
        """
        Execute the time integration to solve the Schrödinger equation.
        
        Parameters:
        - store_solutions: Whether to store solutions at each time step
        - compute_errors: Whether to compute L2 errors (requires analytical solution)
        
        Returns:
        - Dictionary with times, solutions (optional), and errors (optional)
        """
        print("Starting time integration...")
        
        # Initialize with initial condition
        self.u_previous = self._create_initial_solution()
        
        if store_solutions:
            self.solutions = [self.u_previous.copy()]
        
        if compute_errors:
            self.errors = [self._compute_l2_error(0.0)]
        
        # Time stepping loop with tqdm progress bar
        for n in tqdm.tqdm(range(1, self.N_time + 1), desc="Time integration", unit="step"):
            t_current = n * self.dt
            
            # Solve current time step
            self.u_current = self._solve_time_step(t_current)
            
            # Store solution if requested
            if store_solutions:
                self.solutions.append(self.u_current.copy())
            
            # Compute error if requested
            if compute_errors:
                error = self._compute_l2_error(t_current)
                self.errors.append(error)
                # Update progress bar description with current error
                if n % max(1, self.N_time // 10) == 0:
                    tqdm.tqdm.write(f"t={t_current:.4f}, L2 error={error:.6e}")
            
            # Update for next iteration
            self.u_previous.x.array[:] = self.u_current.x.array[:]
        
        print("\nTime integration completed!")
        
        # Return results
        results = {'times': self.times}
        if store_solutions:
            results['solutions'] = self.solutions
        if compute_errors:
            results['errors'] = np.array(self.errors)
        
        return results
    
    def _compute_l2_error(self, t: float) -> float:
        """Compute L2 error against analytical solution."""
        if self.analytical_solution is None:
            return 0.0
        
        # Create analytical solution at current time
        u_exact = fem.Function(self.V, dtype=np.complex128)
        u_exact.interpolate(lambda x: self.analytical_solution(x, t))
        
        # Compute error
        if t == 0:
            u_numerical = self.u_previous
        else:
            u_numerical = self.u_current
        
        error_func = u_numerical - u_exact
        l2_error = np.sqrt(fem.assemble_scalar(fem.form(
            ufl.inner(error_func, error_func) * ufl.dx
        ))).real
        
        return l2_error
    
    def get_solution_at_time(self, t: float):
        """Get solution at a specific time (requires stored solutions)."""
        if not self.solutions:
            raise ValueError("No solutions stored. Run solve() with store_solutions=True")
        
        # Find closest time index
        time_idx = np.argmin(np.abs(self.times - t))
        return self.solutions[time_idx]
    
    def plot_solution(self, t: float, plot_type: str = 'both', save_path: Optional[str] = None, plot_3d: bool = False):
        """
        Plot the numerical and analytical solutions at a given time using matplotlib.
        
        Parameters:
        - t: Time at which to plot the solution
        - plot_type: 'both', 'real', 'imag', or 'abs'
        - save_path: Optional path to save the plot (will be placed in figures/ directory)
        - plot_3d: If True, create a 3D surface plot
        """
        # Get solution at time t
        u_solution = self.get_solution_at_time(t)
        
        # Get coordinates of degrees of freedom
        X = self.V.tabulate_dof_coordinates()
        
        # Create analytical solution if available
        if self.analytical_solution is not None:
            u_exact = fem.Function(self.V, dtype=np.complex128)
            u_exact.interpolate(lambda x: self.analytical_solution(x, t))
        
        if plot_3d:
            # Create 3D surface plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            if plot_type == 'real':
                values_num = u_solution.x.array.real
                values_exact = u_exact.x.array.real if self.analytical_solution is not None else None
                title = 'Real'
            elif plot_type == 'imag':
                values_num = u_solution.x.array.imag
                values_exact = u_exact.x.array.imag if self.analytical_solution is not None else None
                title = 'Imaginary'
            else:  # abs
                values_num = np.abs(u_solution.x.array)
                values_exact = np.abs(u_exact.x.array) if self.analytical_solution is not None else None
                title = 'Absolute'
            
            # Plot numerical solution
            surf_num = ax.plot_trisurf(X[:, 0], X[:, 1], values_num, 
                                     color='royalblue', linewidth=0, alpha=0.5,
                                     label='Numerical')
            
            # Plot analytical solution if available
            if self.analytical_solution is not None:
                surf_exact = ax.plot_trisurf(X[:, 0], X[:, 1], values_exact,
                                           color='crimson', linewidth=0, alpha=0.5,
                                           label='Analytical')
            
            ax.set_title(f'{title} part at t={t:.4f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            ax.legend()
            
            if save_path is None:
                save_path = f'figures/solution_3d_{plot_type}_t{t:.4f}.svg'
            
        else:  # 2D plots
            if plot_type == 'both':
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # Real part - Numerical
                scatter1 = ax1.scatter(X[:, 0], X[:, 1], 
                                     c=u_solution.x.array.real,
                                     cmap="viridis")
                ax1.set_title(f'Real part (Numerical) at t={t:.4f}')
                plt.colorbar(scatter1, ax=ax1)
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                
                # Imaginary part - Numerical
                scatter2 = ax2.scatter(X[:, 0], X[:, 1], 
                                     c=u_solution.x.array.imag,
                                     cmap="viridis")
                ax2.set_title(f'Imaginary part (Numerical) at t={t:.4f}')
                plt.colorbar(scatter2, ax=ax2)
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                
                if self.analytical_solution is not None:
                    # Real part - Analytical
                    scatter3 = ax3.scatter(X[:, 0], X[:, 1],
                                         c=u_exact.x.array.real,
                                         cmap="viridis")
                    ax3.set_title(f'Real part (Analytical) at t={t:.4f}')
                    plt.colorbar(scatter3, ax=ax3)
                    ax3.set_xlabel('x')
                    ax3.set_ylabel('y')
                    
                    # Imaginary part - Analytical
                    scatter4 = ax4.scatter(X[:, 0], X[:, 1],
                                         c=u_exact.x.array.imag,
                                         cmap="viridis")
                    ax4.set_title(f'Imaginary part (Analytical) at t={t:.4f}')
                    plt.colorbar(scatter4, ax=ax4)
                    ax4.set_xlabel('x')
                    ax4.set_ylabel('y')
                
                plt.tight_layout()
                
                if save_path is None:
                    save_path = f'figures/solution_both_t{t:.4f}.svg'
            else:
                if self.analytical_solution is not None:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                else:
                    fig, ax1 = plt.subplots(figsize=(8, 6))
                    ax2 = None
                
                if plot_type == 'real':
                    values_num = u_solution.x.array.real
                    values_exact = u_exact.x.array.real if self.analytical_solution is not None else None
                    title = 'Real'
                elif plot_type == 'imag':
                    values_num = u_solution.x.array.imag
                    values_exact = u_exact.x.array.imag if self.analytical_solution is not None else None
                    title = 'Imaginary'
                else:  # abs
                    values_num = np.abs(u_solution.x.array)
                    values_exact = np.abs(u_exact.x.array) if self.analytical_solution is not None else None
                    title = 'Absolute'
                
                # Numerical solution
                scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=values_num, cmap="viridis")
                plt.colorbar(scatter1, ax=ax1)
                ax1.set_title(f'{title} part (Numerical) at t={t:.4f}')
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                
                # Analytical solution
                if self.analytical_solution is not None:
                    scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=values_exact, cmap="viridis")
                    plt.colorbar(scatter2, ax=ax2)
                    ax2.set_title(f'{title} part (Analytical) at t={t:.4f}')
                    ax2.set_xlabel('x')
                    ax2.set_ylabel('y')
                
                if save_path is None:
                    save_path = f'figures/solution_{plot_type}_t{t:.4f}.svg'
        
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Solution plot saved to: {save_path}")

    def animate_solution(self, plot_type: str = 'abs', plot_3d: bool = False, fps: int = 1, num_max_frames: int = 11):
        """
        Create an animation of the solution evolution over time.
        
        Parameters:
        - plot_type: 'real', 'imag', or 'abs'
        - plot_3d: If True, create a 3D surface animation
        - fps: Frames per second for the animation
        
        Note: If there are more than 100 time steps, the animation will use
        100 evenly distributed frames across the time interval.
        """
        if not self.solutions:
            raise ValueError("No solutions stored. Run solve() with store_solutions=True")
            
        import matplotlib.animation as animation
        
        # Get coordinates of degrees of freedom
        X = self.V.tabulate_dof_coordinates()
        
        # If we have more than 100 time steps, downsample to 100 frames
        n_frames = min(num_max_frames, len(self.solutions))
        if len(self.solutions) > num_max_frames:
            # Calculate indices for evenly spaced frames
            frame_indices = np.linspace(0, len(self.solutions) - 1, n_frames, dtype=int)
            solutions_subset = [self.solutions[i] for i in frame_indices]
            times_subset = self.times[frame_indices]
        else:
            solutions_subset = self.solutions
            times_subset = self.times
            frame_indices = np.arange(len(self.solutions))
            
        # Create figure and axis
        if plot_3d:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            if self.analytical_solution is not None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            else:
                fig, ax1 = plt.subplots(figsize=(8, 8))
                ax2 = None
            
        # Function to get values based on plot type
        def get_values(solution, is_analytical=False, current_frame=0):
            if is_analytical:
                u_exact = fem.Function(self.V, dtype=np.complex128)
                u_exact.interpolate(lambda x: self.analytical_solution(x, times_subset[current_frame]))
                solution = u_exact
                
            if plot_type == 'real':
                return solution.x.array.real
            elif plot_type == 'imag':
                return solution.x.array.imag
            else:  # abs
                return np.abs(solution.x.array)
                
        # Get value range for consistent z-axis limits
        vmin = min(get_values(sol).min() for sol in solutions_subset)
        vmax = max(get_values(sol).max() for sol in solutions_subset)
        if self.analytical_solution is not None:
            vmin_exact = min(get_values(None, True, i).min() for i in range(n_frames))
            vmax_exact = max(get_values(None, True, i).max() for i in range(n_frames))
            vmin = min(vmin, vmin_exact)
            vmax = max(vmax, vmax_exact)
        
        if plot_3d:
            # Initial surface plots
            surf_num = ax.plot_trisurf(X[:, 0], X[:, 1], get_values(solutions_subset[0]),
                                     color='crimson', linewidth=0, alpha=0.5,
                                     label='Numerical')
            if self.analytical_solution is not None:
                surf_exact = ax.plot_trisurf(X[:, 0], X[:, 1], get_values(None, True, 0),
                                           color='royalblue', linewidth=0, alpha=0.5,
                                           label='Analytical')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            ax.legend()
            
            def update(frame):
                ax.clear()
                values_num = get_values(solutions_subset[frame])
                surf_num = ax.plot_trisurf(X[:, 0], X[:, 1], values_num,
                                         color='crimson', linewidth=0, alpha=0.5,
                                         label='Numerical')
                if self.analytical_solution is not None:
                    values_exact = get_values(None, True, frame)
                    surf_exact = ax.plot_trisurf(X[:, 0], X[:, 1], values_exact,
                                               color='royalblue', linewidth=0, alpha=0.5,
                                               label='Analytical')
                ax.set_title(f't = {times_subset[frame]:.4f}')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('u')
                ax.set_zlim(vmin, vmax)
                ax.legend()
                return surf_num,
        else:
            # Initial scatter plots
            scatter_num = ax1.scatter(X[:, 0], X[:, 1], c=get_values(solutions_subset[0]),
                                    cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar(scatter_num, ax=ax1)
            ax1.set_title('Numerical Solution')
            
            if self.analytical_solution is not None:
                scatter_exact = ax2.scatter(X[:, 0], X[:, 1], c=get_values(None, True, 0),
                                          cmap='viridis', vmin=vmin, vmax=vmax)
                plt.colorbar(scatter_exact, ax=ax2)
                ax2.set_title('Analytical Solution')
            
            def update(frame):
                ax1.clear()
                values_num = get_values(solutions_subset[frame])
                scatter_num = ax1.scatter(X[:, 0], X[:, 1], c=values_num,
                                        cmap='viridis', vmin=vmin, vmax=vmax)
                ax1.set_title(f'Numerical Solution (t = {times_subset[frame]:.4f})')
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                
                if self.analytical_solution is not None:
                    ax2.clear()
                    values_exact = get_values(None, True, frame)
                    scatter_exact = ax2.scatter(X[:, 0], X[:, 1], c=values_exact,
                                              cmap='viridis', vmin=vmin, vmax=vmax)
                    ax2.set_title(f'Analytical Solution (t = {times_subset[frame]:.4f})')
                    ax2.set_xlabel('x')
                    ax2.set_ylabel('y')
                
                return scatter_num,
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                     interval=1000/fps, blit=True)
        
        # Save animation
        save_path = f'figures/solution_animation_{plot_type}_{"3d" if plot_3d else "2d"}.gif'
        anim.save(save_path, writer='pillow', fps=fps)
        plt.close()
        print(f"Animation saved to: {save_path} ({n_frames} frames)")
        if len(self.solutions) > num_max_frames:
            print(f"Note: Time steps were downsampled to {num_max_frames} frames for faster animation.")
    
    def plot_error_evolution(self, save_path: Optional[str] = None):
        """Plot the L2 error evolution over time."""
        if not self.errors:
            raise ValueError("No errors computed. Run solve() with compute_errors=True")
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(self.times, self.errors, 'b-', linewidth=2, label='L2 Error')
        plt.xlabel('Time')
        plt.ylabel('L2 Error')
        plt.title('Error Evolution Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path is None:
            save_path = 'figures/error_evolution.svg'
        
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Error evolution plot saved to: {save_path}")
    
   


def run_example():
    """Run example simulation and create visualizations."""
    print("\nRunning Schrödinger equation example...")

    def step_potential(x, t):
        return t * 0.2 if (x[0] < 0.5 and x[1] < 0.5) else 0

    def harmonic_potential(x):
        return 1 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2)

    solver = StationarySchrodingerSolver(potential=harmonic_potential)
    phi0 = solver.solve_ground_state()
    solver.plot_2d(save_path='figures/ground_state_2d.png')
    solver.plot_3d(save_path='figures/ground_state_3d.png')
    
    # Create solver with default settings
    solver = SchrodingerSolver(
        nx=64, 
        ny=64,
        T_final=2.0, 
        N_time=100000,
        # potential=step_potential,
        # initial_condition=phi0,
        analytical_solution=None
    )
    
    # Solve the equation
    results = solver.solve(store_solutions=True, compute_errors=True)
    
    # Plot error evolution
    solver.plot_error_evolution()
    
    # Plot initial state
    # solver.plot_solution(0.0, plot_type='both')
    
    # Plot final state in 2D and 3D
    solver.plot_solution(solver.T_final, plot_type='abs')
    solver.plot_solution(solver.T_final, plot_type='abs', plot_3d=True)
    
    # Create animations
    solver.animate_solution(plot_type='abs', plot_3d=False)
    solver.animate_solution(plot_type='abs', plot_3d=True)
    
    return solver


if __name__ == "__main__":
    # Run example
    solver = run_example()
    
