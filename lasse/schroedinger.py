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
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl
from petsc4py import PETSc
from mpi4py import MPI
import matplotlib.pyplot as plt
from typing import Callable, Optional, Union, Tuple, List, Any
from abc import ABC, abstractmethod
from potentials import ModelPotential

def get_model_potential():
    from potentials import ModelPotential
    V = ModelPotential(
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
    # Wrap the potential to change signature from (x,y,t) to (x,t)
    def wrapped_potential(x, t=0.0):
        print(np.max(x))
        print(np.min(x))
        print(t)
        return V(x[:,0], x[:,1], t)
    return wrapped_potential


class Potential:
    def __init__(self, t=0.0):
        self.t = t
        self.f = ModelPotential(
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

    def __call__(self, x):
        return self.f(x[0] + self.t)  # custom logic

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
        
        # Set potential (default to zero potential)
        if potential is None:
            try:
                self.potential = HarmonicPotential()
            except:
                self.potential = None
        else:
            self.potential = potential
            
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

        # Potential form
        if self.potential is not None:
            Potential = dolfinx.fem.Function(self.V)
            Potential.interpolate(self.potential(t=t_current))
        
        # Constants for complex arithmetic
        i_dt = fem.Constant(self.mesh, np.complex128(1j / self.dt))
        
        # Bilinear form: (i/dt)⟨u,v⟩ + ⟨∇u,∇v⟩
        a = i_dt * ufl.inner(u, v) * ufl.dx - ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        
        # Linear form: (i/dt)⟨u_prev,v⟩
        if self.potential is not None:
            L = i_dt * ufl.inner(self.u_previous, v) * ufl.dx + ufl.inner(Potential, self.u_previous) * ufl.dx
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
        
        # Time stepping loop
        for n in range(1, self.N_time + 1):
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
            
            # Update for next iteration
            self.u_previous.x.array[:] = self.u_current.x.array[:]
            
            # Progress reporting
            if n % max(1, self.N_time // 10) == 0:
                progress = 100 * n / self.N_time
                if compute_errors:
                    print(f"  Progress: {progress:5.1f}% (t={t_current:.4f}, L2 error={self.errors[-1]:.6e})")
                else:
                    print(f"  Progress: {progress:5.1f}% (t={t_current:.4f})")
        
        print("Time integration completed!")
        
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
        Plot the solution at a given time using matplotlib.
        
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
        
        if plot_3d:
            # Create 3D surface plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            if plot_type == 'real':
                values = u_solution.x.array.real
                title = 'Real'
            elif plot_type == 'imag':
                values = u_solution.x.array.imag
                title = 'Imaginary'
            else:  # abs
                values = np.abs(u_solution.x.array)
                title = 'Absolute'
            
            surf = ax.plot_trisurf(X[:, 0], X[:, 1], values, 
                                 cmap='viridis', linewidth=0)
            plt.colorbar(surf, ax=ax)
            ax.set_title(f'{title} part at t={t:.4f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            
            if save_path is None:
                save_path = f'figures/solution_3d_{plot_type}_t{t:.4f}.svg'
            
        else:  # 2D plots
            if plot_type == 'both':
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Real part
                scatter1 = ax1.scatter(X[:, 0], X[:, 1], 
                                     c=u_solution.x.array.real,
                                     cmap="viridis")
                ax1.set_title(f'Real part at t={t:.4f}')
                plt.colorbar(scatter1, ax=ax1)
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                
                # Imaginary part
                scatter2 = ax2.scatter(X[:, 0], X[:, 1], 
                                     c=u_solution.x.array.imag,
                                     cmap="viridis")
                ax2.set_title(f'Imaginary part at t={t:.4f}')
                plt.colorbar(scatter2, ax=ax2)
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                
                plt.tight_layout()
                
                if save_path is None:
                    save_path = f'figures/solution_both_t{t:.4f}.svg'
            else:
                plt.figure(figsize=(8, 6))
                
                if plot_type == 'real':
                    values = u_solution.x.array.real
                    title = 'Real'
                elif plot_type == 'imag':
                    values = u_solution.x.array.imag
                    title = 'Imaginary'
                else:  # abs
                    values = np.abs(u_solution.x.array)
                    title = 'Absolute'
                
                scatter = plt.scatter(X[:, 0], X[:, 1], c=values, cmap="viridis")
                plt.colorbar(scatter)
                plt.title(f'{title} part at t={t:.4f}')
                plt.xlabel('x')
                plt.ylabel('y')
                
                if save_path is None:
                    save_path = f'figures/solution_{plot_type}_t{t:.4f}.svg'
        
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Solution plot saved to: {save_path}")

    def animate_solution(self, plot_type: str = 'abs', plot_3d: bool = False, fps: int = 10):
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
        n_frames = min(100, len(self.solutions))
        if len(self.solutions) > 100:
            # Calculate indices for evenly spaced frames
            frame_indices = np.linspace(0, len(self.solutions) - 1, n_frames, dtype=int)
            solutions_subset = [self.solutions[i] for i in frame_indices]
            times_subset = self.times[frame_indices]
        else:
            solutions_subset = self.solutions
            times_subset = self.times
            
        # Create figure and axis
        if plot_3d:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            
        # Function to get values based on plot type
        def get_values(solution):
            if plot_type == 'real':
                return solution.x.array.real
            elif plot_type == 'imag':
                return solution.x.array.imag
            else:  # abs
                return np.abs(solution.x.array)
                
        # Get value range for consistent colormap
        vmin = min(get_values(sol).min() for sol in solutions_subset)
        vmax = max(get_values(sol).max() for sol in solutions_subset)
        
        if plot_3d:
            # Initial surface plot
            surf = ax.plot_trisurf(X[:, 0], X[:, 1], get_values(solutions_subset[0]),
                                 cmap='viridis', linewidth=0, vmin=vmin, vmax=vmax)
            plt.colorbar(surf, ax=ax)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            
            def update(frame):
                ax.clear()
                values = get_values(solutions_subset[frame])
                surf = ax.plot_trisurf(X[:, 0], X[:, 1], values,
                                     cmap='viridis', linewidth=0, vmin=vmin, vmax=vmax)
                ax.set_title(f't = {times_subset[frame]:.4f}')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('u')
                ax.set_zlim(vmin, vmax)
                return surf,
        else:
            # Initial scatter plot
            scatter = ax.scatter(X[:, 0], X[:, 1], c=get_values(solutions_subset[0]),
                               cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar(scatter, ax=ax)
            
            def update(frame):
                ax.clear()
                values = get_values(solutions_subset[frame])
                scatter = ax.scatter(X[:, 0], X[:, 1], c=values,
                                   cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_title(f't = {times_subset[frame]:.4f}')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                return scatter,
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                     interval=1000/fps, blit=True)
        
        # Save animation
        save_path = f'figures/solution_animation_{plot_type}_{"3d" if plot_3d else "2d"}.gif'
        anim.save(save_path, writer='pillow', fps=fps)
        plt.close()
        print(f"Animation saved to: {save_path} ({n_frames} frames)")
        if len(self.solutions) > 100:
            print("Note: Time steps were downsampled to 100 frames for faster animation.")
    
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

    potential = Potential()
    
    # Create solver with default settings
    solver = SchrodingerSolver(
        nx=64, 
        ny=64,
        T_final=1.0, 
        N_time=10000,
        potential=potential
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
    solver.animate_solution(plot_type='imag', plot_3d=False)
    solver.animate_solution(plot_type='imag', plot_3d=True)
    
    return solver


if __name__ == "__main__":
    # Run example
    solver = run_example()
    
