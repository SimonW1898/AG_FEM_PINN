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
import os
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

# Global plotting configuration
WAVEFUNCTION_COLOR = 'royalblue'
WAVEFUNCTION_ALPHA = 0.9
POTENTIAL_CMAP = 'viridis'
POTENTIAL_ALPHA = 0.9
WAVEFUNCTION_2D_CMAP = 'viridis'
INITIAL_SOLUTION_COLOR = 'darkorange'

# Font size configuration for consistency
PLOT_TITLE_FONTSIZE = 14
PLOT_LABEL_FONTSIZE = 12
PLOT_TICK_FONTSIZE = 10

# Configure matplotlib for consistent font sizes
plt.rcParams.update({
    'font.size': PLOT_TICK_FONTSIZE,
    'axes.titlesize': PLOT_TITLE_FONTSIZE,
    'axes.labelsize': PLOT_LABEL_FONTSIZE,
    'xtick.labelsize': PLOT_TICK_FONTSIZE,
    'ytick.labelsize': PLOT_TICK_FONTSIZE,
    'legend.fontsize': PLOT_TICK_FONTSIZE
})


class StationarySchrodingerSolver:
    """
    Solves the time-independent Schrödinger equation:
        -Δφ + V(x, y) φ = E φ
    on the unit square with Dirichlet BCs.
    """

    def __init__(self,
                 potential: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 nx: int = 16,
                 ny: int = 16):
        self.nx = nx
        self.ny = ny
        self.mesh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
        self.V_space = fem.functionspace(self.mesh, ("Lagrange", 1))

        # Initialize potential function
        self.V_func = fem.Function(self.V_space)
        if potential is not None:
            self.V_func.interpolate(potential)
        else:
            # Set to zero potential
            self.V_func.x.array[:] = 0.0

        self.A, self.M = self._assemble_operators()
        self.ground_state = None  # to be computed later
        self.eigenvalues = None
        self.eigenfunctions = None

    def _assemble_operators(self):
        u = ufl.TrialFunction(self.V_space)
        v = ufl.TestFunction(self.V_space)

        # Stiffness matrix: ∫∇u·∇v dx
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        
        # Add potential term if present
        if self.V_func is not None and np.any(self.V_func.x.array != 0):
            print("Adding potential term")
            a += ufl.inner(self.V_func * u, v) * ufl.dx
            
        # Mass matrix: ∫u·v dx
        m = ufl.inner(u, v) * ufl.dx

        # Assemble matrices with boundary conditions
        A = fem.petsc.assemble_matrix(fem.form(a), bcs=self._get_bcs())
        A.assemble()
        M = fem.petsc.assemble_matrix(fem.form(m), bcs=self._get_bcs())
        M.assemble()

        return A, M

    def _get_bcs(self):
        # Create connectivity for boundary facets
        tdim = self.mesh.topology.dim
        fdim = tdim - 1
        self.mesh.topology.create_connectivity(fdim, tdim)
        
        # Get boundary facets
        boundary_facets = mesh.exterior_facet_indices(self.mesh.topology)
        
        # Locate boundary DOFs
        boundary_dofs = fem.locate_dofs_topological(self.V_space, fdim, boundary_facets)
        
        # Create zero Dirichlet boundary condition
        uD = fem.Function(self.V_space)
        uD.interpolate(lambda x: np.zeros(x.shape[1]))
        
        return [fem.dirichletbc(uD, boundary_dofs)]

    def get_filtered_ground_state(self):
        """
        Get the ground state by filtering out the spurious unit eigenvalue.
        Uses solve_eigenvalues and selects the smallest non-unit eigenvalue.
        """
        # Solve for multiple eigenvalues first
        if self.eigenvalues is None:
            self.solve_eigenvalues(n_eigenvalues=10)
        
        # Find the smallest eigenvalue that is not 1.0 (spurious)
        ground_state_idx = None
        for i, eig_val in enumerate(self.eigenvalues):
            if abs(eig_val - 1.0) > 0.1:  # Not the spurious eigenvalue
                ground_state_idx = i
                break
        
        if ground_state_idx is None:
            raise RuntimeError("No non-spurious eigenvalues found")
        
        # Set the ground state
        self.ground_state = self.eigenfunctions[ground_state_idx]
        self.ground_state_eigenvalue = self.eigenvalues[ground_state_idx]
        
        if MPI.COMM_WORLD.rank == 0:
            print(f"Selected ground state (mode {ground_state_idx}): {self.ground_state_eigenvalue:.6f}")
        
        return self.ground_state

    def plot_filtered_ground_state_2d(self, plot_type='abs', title=None, save_path=None, show=True):
        """Plot the filtered ground state in 2D."""
        if self.ground_state is None:
            self.get_filtered_ground_state()
        
        if title is None:
            title = f'Filtered Ground State |φ|² (E={self.ground_state_eigenvalue:.4f})'
        
        self.plot_2d(plot_type=plot_type, title=title, save_path=save_path, show=show)

    def plot_filtered_ground_state_3d(self, plot_type='abs', title=None, save_path=None, show=True):
        """Plot the filtered ground state in 3D."""
        if self.ground_state is None:
            self.get_filtered_ground_state()
        
        if title is None:
            title = f'Filtered Ground State (3D) |φ|² (E={self.ground_state_eigenvalue:.4f})'
        
        self.plot_3d(plot_type=plot_type, title=title, save_path=save_path, show=show)

    def get_ground_state_as_initial_condition(self):
        """
        Get the filtered ground state nodal values for initial condition.
        Returns the array of values at the current DOFs.
        """
        if self.ground_state is None:
            self.get_filtered_ground_state()
        
        # Return the nodal values directly
        return self.ground_state.x.array.copy()

    def solve_eigenvalues(self, n_eigenvalues=10, max_energy=None, tol=1e-8):
        """
        Solve for multiple eigenvalues with robust filtering.
        
        Args:
            n_eigenvalues: Number of eigenvalues to compute
            max_energy: Maximum energy cutoff (removes high-frequency spurious modes)
            tol: Tolerance for eigenvalue solver
        """
        # Create eigenvalue solver
        eps = SLEPc.EPS().create(MPI.COMM_WORLD)
        eps.setOperators(self.A, self.M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        
        # Request more eigenvalues than needed to filter later
        eps.setDimensions(min(n_eigenvalues * 2, self.V_space.dofmap.index_map.size_global))
        eps.setTolerances(tol)
        
        # Set solver options for better performance
        eps.setFromOptions()
        eps.solve()

        # Check convergence
        n_conv = eps.getConverged()
        if n_conv < 1:
            raise RuntimeError("No eigenpairs found.")

        # Extract eigenvalues and eigenvectors
        eigenvalues = []
        eigenvectors = []
        
        for i in range(n_conv):
            eig_val = eps.getEigenvalue(i)
            if eig_val.imag != 0:
                continue  # Skip complex eigenvalues
            
            eig_val = eig_val.real
            
            # Apply energy cutoff
            if max_energy is not None and eig_val > max_energy:
                continue
                
            r, _ = self.A.getVecs()
            eps.getEigenvector(i, r)
            
            eigenvalues.append(eig_val)
            eigenvectors.append(r.getArray().copy())
            
            if len(eigenvalues) >= n_eigenvalues:
                break

        if MPI.COMM_WORLD.rank == 0:
            print(f"Found {len(eigenvalues)} valid eigenvalues out of {n_conv} converged")
            print("Eigenvalues:")
            for i, eig_val in enumerate(eigenvalues[:10]):  # Show first 10
                print(f"  λ_{i}: {eig_val:.6f}")

        # Create eigenfunction objects
        eigenfunctions = []
        for i, eigvec in enumerate(eigenvectors):
            phi = fem.Function(self.V_space)
            phi.x.array[:] = eigvec
            
            # Normalize
            norm_sq = fem.assemble_scalar(fem.form(ufl.inner(phi, phi) * ufl.dx))
            if np.abs(norm_sq) > 0:
                phi.x.array[:] /= np.sqrt(norm_sq)
            
            eigenfunctions.append(phi)

        self.eigenvalues = eigenvalues
        self.eigenfunctions = eigenfunctions
        return eigenvalues, eigenfunctions

    def estimate_energy_cutoff(self, safety_factor=0.1):
        """
        Estimate reasonable energy cutoff based on grid spacing.
        
        Args:
            safety_factor: Fraction of max representable energy to use as cutoff
        """
        # Estimate based on grid spacing
        h = 1.0 / max(self.nx, self.ny)  # Approximate grid spacing
        max_representable_energy = (np.pi / h) ** 2  # Nyquist frequency squared
        return safety_factor * max_representable_energy

    def plot_eigenfunction(self, mode_index: int, plot_type: str = 'real', 
                          save_path: Optional[str] = None, show: bool = False, title: Optional[str] = None):
        """Plot a single eigenfunction."""
        if self.eigenfunctions is None:
            raise ValueError("No eigenfunctions computed. Run solve_eigenvalues() first.")
        
        if mode_index >= len(self.eigenfunctions):
            raise ValueError(f"Mode index {mode_index} out of range. Only {len(self.eigenfunctions)} modes available.")
        
        # Get coordinates
        X = self.V_space.tabulate_dof_coordinates()
        
        phi = self.eigenfunctions[mode_index]
        eig_val = self.eigenvalues[mode_index]
        
        if title is None:
            title = f'Mode {mode_index}, E = {eig_val:.4f}'
        
        if plot_type == 'real':
            Z = phi.x.array.real
            label = 'Re(u)'
        elif plot_type == 'imag':
            Z = phi.x.array.imag
            label = 'Im(u)'
        elif plot_type == 'abs':
            Z = np.abs(phi.x.array)**2
            label = '|u|²'
        else:
            raise ValueError("plot_type must be 'real', 'imag', or 'abs'")
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=Z, cmap=WAVEFUNCTION_2D_CMAP, s=20)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(scatter, ax=ax, label=label)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_eigenvalue_spectrum(self, max_modes=20, save_path=None, show=True):
        """Plot the eigenvalue spectrum."""
        if self.eigenvalues is None:
            raise RuntimeError("Must solve eigenvalues first")
            
        n_plot = min(len(self.eigenvalues), max_modes)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_plot), self.eigenvalues[:n_plot], 'bo-', markersize=6)
        plt.xlabel('Mode Index')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalue Spectrum')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        plt.close()

    def plot_all_eigenfunctions(self, n_modes=6, plot_type='abs', save_path=None, show=True):
        """Plot multiple eigenfunctions in a grid."""
        if self.eigenfunctions is None:
            raise RuntimeError("Must solve eigenvalues first")
            
        n_plot = min(n_modes, len(self.eigenfunctions))
        
        # Calculate grid dimensions
        cols = 3
        rows = (n_plot + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if n_plot == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        X = self.V_space.tabulate_dof_coordinates()
        
        for i in range(n_plot):
            phi = self.eigenfunctions[i]
            eig_val = self.eigenvalues[i]
            
            if plot_type == 'real':
                Z = phi.x.array.real
                label = 'Re(u)'
            elif plot_type == 'imag':
                Z = phi.x.array.imag
                label = 'Im(u)'
            else:
                Z = np.abs(phi.x.array)**2
                label = '|u|²'
            
            sc = axes[i].scatter(X[:, 0], X[:, 1], c=Z, cmap=WAVEFUNCTION_2D_CMAP, s=8)
            axes[i].set_title(f'Mode {i}, E = {eig_val:.4f}')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            axes[i].axis('equal')
            plt.colorbar(sc, ax=axes[i], label=label)
        
        # Hide unused subplots
        for i in range(n_plot, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def plot_2d(self, plot_type='abs', title='Ground state', save_path=None, show=True):
        if self.ground_state is None:
            raise RuntimeError("Call solve_ground_state() first.")

        X = self.V_space.tabulate_dof_coordinates()
        if plot_type == 'real':
            Z = self.ground_state.x.array.real
            label = 'Re(u)'
        elif plot_type == 'imag':
            Z = self.ground_state.x.array.imag
            label = 'Im(u)'
        else:
            Z = np.abs(self.ground_state.x.array)**2
            label = '|u|²'

        plt.figure(figsize=(6, 5))
        sc = plt.scatter(X[:, 0], X[:, 1], c=Z, cmap=WAVEFUNCTION_2D_CMAP, s=10)
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
        plt.close()

    def plot_3d(self, plot_type='abs', title='Ground state (3D)', save_path=None, show=True):
        if self.ground_state is None:
            raise RuntimeError("Call solve_ground_state() first.")

        X = self.V_space.tabulate_dof_coordinates()
        if plot_type == 'real':
            Z = self.ground_state.x.array.real
            label = 'Re(u)'
        elif plot_type == 'imag':
            Z = self.ground_state.x.array.imag
            label = 'Im(u)'
        else:
            Z = np.abs(self.ground_state.x.array)**2
            label = '|u|²'

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(X[:, 0], X[:, 1], Z, color=WAVEFUNCTION_COLOR, linewidth=0, alpha=WAVEFUNCTION_ALPHA)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(label)
        ax.view_init(elev=30, azim=45)
        # Adjust layout for 3D plots using set_box_aspect for better control
        ax.set_box_aspect(None, zoom=0.85)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_eigenfunction_3d(self, mode_index: int, plot_type: str = 'real', 
                             save_path: Optional[str] = None, show: bool = False, title: Optional[str] = None):
        """Plot a single eigenfunction in 3D."""
        if self.eigenfunctions is None:
            raise ValueError("No eigenfunctions computed. Run solve_eigenvalues() first.")
        
        if mode_index >= len(self.eigenfunctions):
            raise ValueError(f"Mode index {mode_index} out of range. Only {len(self.eigenfunctions)} modes available.")
        
        # Get coordinates
        X = self.V_space.tabulate_dof_coordinates()
        
        phi = self.eigenfunctions[mode_index]
        eig_val = self.eigenvalues[mode_index]
        
        if title is None:
            title = f'Mode {mode_index}, E = {eig_val:.4f}'
        
        if plot_type == 'real':
            Z = phi.x.array.real
            label = 'Re(u)'
        elif plot_type == 'imag':
            Z = phi.x.array.imag
            label = 'Im(u)'
        elif plot_type == 'abs':
            Z = np.abs(phi.x.array)**2
            label = '|u|²'
        else:
            raise ValueError("plot_type must be 'real', 'imag', or 'abs'")
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(X[:, 0], X[:, 1], Z, color=WAVEFUNCTION_COLOR, linewidth=0, alpha=WAVEFUNCTION_ALPHA)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(label)
        ax.view_init(elev=30, azim=45)
        # Adjust layout for 3D plots using set_box_aspect for better control
        ax.set_box_aspect(None, zoom=0.85)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()


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
                 dt: float = 0.01,
                 potential: Optional[Any] = None,
                 initial_condition: Optional[Callable] = None,
                 analytical_solution: Optional[Callable] = None):
        """
        Initialize the Schrödinger equation solver.
        
        Parameters:
        - nx, ny: Number of grid points in x and y directions
        - T_final: Final time for simulation
        - dt: Time step size
        - potential: Potential function (Potential class instance)
        - initial_condition: Initial condition function u₀(x,y)
        - analytical_solution: Analytical solution function u(x,y,t) for error analysis
        """
        # Grid and time parameters
        self.nx = int(nx)
        self.ny = int(ny)
        self.T_final = T_final
        self.dt = dt
        self.N_time = int(T_final / dt)
        
        # Create mesh and function space
        self.mesh = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, self.nx, self.ny, dolfinx.mesh.CellType.triangle
        )
        self.V = fem.functionspace(self.mesh, ("Lagrange", 1))
        
        # Set potential
        if potential is not None:
            self.potential = potential
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
            self.analytical_solution = None
        elif analytical_solution == "exact":
            self.analytical_solution = lambda x, t: (
                np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * 
                np.exp(-1j * 2 * np.pi**2 * t)
            )
        else:
            self.analytical_solution = analytical_solution
        
        # Initialize solution storage
        self.solutions: List[Any] = []  # Store solutions at each time step
        self.times = np.linspace(0, T_final, self.N_time + 1)
        self.errors: List[float] = []  # Store L2 errors if analytical solution is available
        
        # Set up boundary conditions
        self._setup_boundary_conditions()
        
        # Initialize current solution
        self.u_current = None
        self.u_previous = None
        
        print(f"Schrödinger solver initialized:")
        print(f"  Grid: {nx}×{ny}, Time steps: {self.N_time + 1}, dt = {self.dt:.6f}")
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
        
        # Check if initial_condition is a callable function or already values
        if callable(self.initial_condition):
            # It's a function, interpolate it
            u_init.interpolate(self.initial_condition)
        else:
            # It's already an array of values, assign directly
            u_init.x.array[:] = self.initial_condition
        
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
    
    def solve(self, store_solutions: bool = True, compute_errors: bool = True, save_interval: int = 1000, 
              save_frames: bool = False, frame_plot_type: str = 'abs'):
        """
        Execute the time integration to solve the Schrödinger equation.
        
        Parameters:
        - store_solutions: Whether to store solutions at each time step
        - compute_errors: Whether to compute L2 errors (requires analytical solution)
        - save_interval: Interval for saving solutions and computing diagnostics (default: 1 = every step)
        - save_frames: If True, save solution plots at each saved time step to figures/solve_frames/
        - frame_plot_type: Plot type for saved frames ('real', 'imag', 'abs')
        
        Returns:
        - Dictionary with times, solutions (optional), and errors (optional)
        """
        print("Starting time integration...")
        
        # Create frame directory if saving frames
        if save_frames:
            frame_dir = 'figures/solve_frames'
            os.makedirs(frame_dir, exist_ok=True)
        
        # Initialize with initial condition
        self.u_previous = self._create_initial_solution()
        
        # Initialize storage lists
        if store_solutions:
            self.solutions = [self.u_previous.copy()]
        
        if compute_errors:
            self.errors = [self._compute_l2_error(0.0)]
        
        # Store times for saved solutions
        self.saved_times = [0.0]
        
        # Store initial diagnostics
        initial_norm = self._compute_l2_norm(self.u_previous)
        if MPI.COMM_WORLD.rank == 0:
            print(f"Initial L2 norm: {initial_norm:.6f}")
        
        # Save initial frame if requested
        if save_frames:
            frame_path = os.path.join(frame_dir, f'solution_t{0.0:.6f}.png')
            self.plot_solution(0.0, plot_type=frame_plot_type, save_path=frame_path, show=False, is_animation=True)
        
        # Time stepping loop with tqdm progress bar
        for n in tqdm.tqdm(range(1, self.N_time + 1), desc="Time integration", unit="step"):
            t_current = n * self.dt
            
            # Solve current time step
            self.u_current = self._solve_time_step(t_current)
            
            # Store solution and compute diagnostics at specified interval
            if n % save_interval == 0:
                if store_solutions:
                    self.solutions.append(self.u_current.copy())
                
                if compute_errors:
                    error = self._compute_l2_error(t_current)
                    self.errors.append(error)
                
                # Store the current time
                self.saved_times.append(t_current)
                
                # Save frame if requested
                if save_frames:
                    frame_path = os.path.join(frame_dir, f'solution_t{t_current:.6f}.png')
                    self.plot_solution(t_current, plot_type=frame_plot_type, save_path=frame_path, show=False, is_animation=True)
                
                # Compute L2 norm of current solution
                l2_norm = self._compute_l2_norm(self.u_current)
                
                # Update progress bar description with current error and norm
                if compute_errors:
                    tqdm.tqdm.write(f"t={t_current:.4f}, L2 error={error:.6e}, ||u||={l2_norm:.6f}")
                else:
                    tqdm.tqdm.write(f"t={t_current:.4f}, ||u||={l2_norm:.6f}")
            
            # Update for next iteration
            self.u_previous.x.array[:] = self.u_current.x.array[:]
        
        print("\nTime integration completed!")
        
        if save_frames:
            print(f"Solution frames saved to: {frame_dir}")
        
        # Return results
        results = {'times': self.saved_times}
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
    
    def _compute_l2_norm(self, u: fem.Function) -> float:
        """Compute L2 norm of a function."""
        l2_norm = np.sqrt(fem.assemble_scalar(fem.form(
            ufl.inner(u, u) * ufl.dx
        ))).real
        
        return l2_norm
    
    def get_solution_at_time(self, t: float):
        """Get solution at a specific time (requires stored solutions)."""
        if not self.solutions:
            raise ValueError("No solutions stored. Run solve() with store_solutions=True")
        
        # Find closest time index in saved times
        time_idx = np.argmin(np.abs(np.array(self.saved_times) - t))
        return self.solutions[time_idx]
    
    def plot_solution(self, t: float, plot_type: str = 'both', save_path: Optional[str] = None, plot_3d: bool = False, show: bool = True, is_animation: bool = False):
        """
        Plot the numerical and analytical solutions at a given time using matplotlib.
        
        Parameters:
        - t: Time at which to plot the solution
        - plot_type: 'both', 'real', 'imag', or 'abs'
        - save_path: Optional path to save the plot
        - plot_3d: If True, create a 3D surface plot
        - show: Whether to display the plot (default: True)
        - is_animation: If True, don't use special initial solution color (for animations)
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
                values_num = np.abs(u_solution.x.array)**2
                values_exact = np.abs(u_exact.x.array)**2 if self.analytical_solution is not None else None
                title = 'Absolute'
            
            # Calculate z-limits based on initial condition for consistency
            if not hasattr(self, '_plot_zlimits') or self._plot_zlimits is None:
                # Handle both numpy arrays and DOLFINx Functions
                if hasattr(self.initial_condition, 'x'):
                    # DOLFINx Function
                    if plot_type == 'real':
                        first_values = self.initial_condition.x.array.real
                    elif plot_type == 'imag':
                        first_values = self.initial_condition.x.array.imag
                    else:  # abs
                        first_values = np.abs(self.initial_condition.x.array)**2
                else:
                    # numpy array - use first solution instead
                    first_solution = self.get_solution_at_time(0.0)
                    if plot_type == 'real':
                        first_values = first_solution.x.array.real
                    elif plot_type == 'imag':
                        first_values = first_solution.x.array.imag
                    else:  # abs
                        first_values = np.abs(first_solution.x.array)**2
                
                if self.analytical_solution is not None:
                    u_exact_first = fem.Function(self.V, dtype=np.complex128)
                    u_exact_first.interpolate(lambda x: self.analytical_solution(x, 0.0))
                    if plot_type == 'real':
                        first_values_exact = u_exact_first.x.array.real
                    elif plot_type == 'imag':
                        first_values_exact = u_exact_first.x.array.imag
                    else:  # abs
                        first_values_exact = np.abs(u_exact_first.x.array)**2
                    
                    zmin = min(first_values.min(), first_values_exact.min())
                    zmax = max(first_values.max(), first_values_exact.max())
                else:
                    zmin = first_values.min()
                    zmax = first_values.max()
                
                # Add some padding (handle edge cases)
                z_range = zmax - zmin
                if z_range < 1e-10:  # Very small range
                    padding = 0.1 * max(abs(zmin), abs(zmax), 0.1)
                else:
                    padding = 0.1 * z_range
                self._plot_zlimits = (zmin - padding, zmax + padding)
            
            # Use initial solution color for static plots at t=0.0, otherwise use regular wavefunction color
            numerical_color = INITIAL_SOLUTION_COLOR if (np.isclose(t, 0.0) and not is_animation) else WAVEFUNCTION_COLOR
            surf_num = ax.plot_trisurf(X[:, 0], X[:, 1], values_num, 
                                     color=numerical_color, linewidth=0, alpha=WAVEFUNCTION_ALPHA)
            
            # Plot analytical solution if available
            if self.analytical_solution is not None:
                surf_exact = ax.plot_trisurf(X[:, 0], X[:, 1], values_exact,
                                           color='crimson', linewidth=0, alpha=WAVEFUNCTION_ALPHA)
            
            ax.set_title(f't = {t:.4f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('|u|²')
            ax.set_zlim(self._plot_zlimits)
            ax.view_init(elev=30, azim=45)
            
            if save_path is None:
                save_path = f'figures/solution_3d_{plot_type}_t{t:.4f}.png'
            
            # Adjust layout for 3D plots using set_box_aspect for better control
            ax.set_box_aspect(None, zoom=0.85)
            
        else:  # 2D plots
            if plot_type == 'both':
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # Real part - Numerical
                scatter1 = ax1.scatter(X[:, 0], X[:, 1], 
                                     c=u_solution.x.array.real,
                                     cmap=WAVEFUNCTION_2D_CMAP)
                ax1.set_title(f't = {t:.4f}')
                plt.colorbar(scatter1, ax=ax1)
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                
                # Imaginary part - Numerical
                scatter2 = ax2.scatter(X[:, 0], X[:, 1], 
                                     c=u_solution.x.array.imag,
                                     cmap=WAVEFUNCTION_2D_CMAP)
                ax2.set_title(f't = {t:.4f}')
                plt.colorbar(scatter2, ax=ax2)
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                
                if self.analytical_solution is not None:
                    # Real part - Analytical
                    scatter3 = ax3.scatter(X[:, 0], X[:, 1],
                                         c=u_exact.x.array.real,
                                         cmap=WAVEFUNCTION_2D_CMAP)
                    ax3.set_title(f't = {t:.4f}')
                    plt.colorbar(scatter3, ax=ax3)
                    ax3.set_xlabel('x')
                    ax3.set_ylabel('y')
                    
                    # Imaginary part - Analytical
                    scatter4 = ax4.scatter(X[:, 0], X[:, 1],
                                         c=u_exact.x.array.imag,
                                         cmap=WAVEFUNCTION_2D_CMAP)
                    ax4.set_title(f't = {t:.4f}')
                    plt.colorbar(scatter4, ax=ax4)
                    ax4.set_xlabel('x')
                    ax4.set_ylabel('y')
                
                plt.tight_layout()
                
                if save_path is None:
                    save_path = f'figures/solution_both_t{t:.4f}.png'
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
                    values_num = np.abs(u_solution.x.array)**2
                    values_exact = np.abs(u_exact.x.array)**2 if self.analytical_solution is not None else None
                    title = 'Absolute'
                
                # Numerical solution
                scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=values_num, cmap=WAVEFUNCTION_2D_CMAP)
                plt.colorbar(scatter1, ax=ax1)
                ax1.set_title(f't = {t:.4f}')
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                
                # Analytical solution
                if self.analytical_solution is not None:
                    scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=values_exact, cmap=WAVEFUNCTION_2D_CMAP)
                    plt.colorbar(scatter2, ax=ax2)
                    ax2.set_title(f't = {t:.4f}')
                    ax2.set_xlabel('x')
                    ax2.set_ylabel('y')
                
                if save_path is None:
                    save_path = f'figures/solution_{plot_type}_t{t:.4f}.png'
        
        if save_path:
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Solution plot saved to: {save_path}")
        if show:
            plt.show()
        else:
            plt.close()

    def animate_solution(self, plot_type: str = 'abs', plot_3d: bool = False, total_duration: float = 5.0, 
                        save_path: Optional[str] = None, save_frames: bool = False):
        """
        Create an animation of the solution evolution over time.
        
        Parameters:
        - plot_type: 'real', 'imag', or 'abs'
        - plot_3d: If True, create a 3D surface animation
        - total_duration: Total duration of the animation in seconds (default: 5.0)
        - save_path: Optional path to save the animation
        - save_frames: If True, save individual frames to figures/animation_frames/
        """
        if not self.solutions:
            raise ValueError("No solutions stored. Run solve() with store_solutions=True")
            
        import matplotlib.animation as animation
        
        # Create frame directory if saving frames
        if save_frames:
            frame_dir = 'figures/animation_frames'
            os.makedirs(frame_dir, exist_ok=True)
        
        # Get coordinates of degrees of freedom
        X = self.V.tabulate_dof_coordinates()
        
        # Use all saved solutions
        solutions_subset = self.solutions
        times_subset = self.saved_times
        n_frames = len(self.solutions)
        
        # Calculate frame rate to achieve desired total duration
        fps = n_frames / total_duration
        interval_ms = 1000 / fps  # Convert to milliseconds for matplotlib
            
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
                return np.abs(solution.x.array)**2
                
        # Get value range for consistent z-axis limits (use initial condition for consistency)
        # Handle both numpy arrays and DOLFINx Functions
        if hasattr(self.initial_condition, 'x'):
            # DOLFINx Function
            if plot_type == 'real':
                first_values = self.initial_condition.x.array.real
            elif plot_type == 'imag':
                first_values = self.initial_condition.x.array.imag
            else:  # abs
                first_values = np.abs(self.initial_condition.x.array)**2
        else:
            # numpy array - use first solution instead
            first_solution = self.get_solution_at_time(0.0)
            if plot_type == 'real':
                first_values = first_solution.x.array.real
            elif plot_type == 'imag':
                first_values = first_solution.x.array.imag
            else:  # abs
                first_values = np.abs(first_solution.x.array)**2
            
        if self.analytical_solution is not None:
            u_exact_first = fem.Function(self.V, dtype=np.complex128)
            u_exact_first.interpolate(lambda x: self.analytical_solution(x, 0.0))
            if plot_type == 'real':
                first_values_exact = u_exact_first.x.array.real
            elif plot_type == 'imag':
                first_values_exact = u_exact_first.x.array.imag
            else:  # abs
                first_values_exact = np.abs(u_exact_first.x.array)**2
            vmin = min(first_values.min(), first_values_exact.min())
            vmax = max(first_values.max(), first_values_exact.max())
        else:
            vmin = first_values.min()
            vmax = first_values.max()
            
        # Add some padding
        padding = 0.1 * (vmax - vmin)
        vmin -= padding
        vmax += padding
        
        if plot_3d:
            # Initial surface plots
            surf_num = ax.plot_trisurf(X[:, 0], X[:, 1], get_values(solutions_subset[0]),
                                     color=WAVEFUNCTION_COLOR, linewidth=0, alpha=WAVEFUNCTION_ALPHA)
            if self.analytical_solution is not None:
                surf_exact = ax.plot_trisurf(X[:, 0], X[:, 1], get_values(None, True, 0),
                                           color='crimson', linewidth=0, alpha=WAVEFUNCTION_ALPHA)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            if plot_type == 'real':
                ax.set_zlabel('Re(u)')
            elif plot_type == 'imag':
                ax.set_zlabel('Im(u)')
            else:  # abs
                ax.set_zlabel('|u|²')
            ax.set_zlim(vmin, vmax)
            ax.view_init(elev=30, azim=45)
            
            def update(frame):
                ax.clear()
                values_num = get_values(solutions_subset[frame])
                surf_num = ax.plot_trisurf(X[:, 0], X[:, 1], values_num,
                                         color=WAVEFUNCTION_COLOR, linewidth=0, alpha=WAVEFUNCTION_ALPHA)
                if self.analytical_solution is not None:
                    values_exact = get_values(None, True, frame)
                    surf_exact = ax.plot_trisurf(X[:, 0], X[:, 1], values_exact,
                                               color='crimson', linewidth=0, alpha=WAVEFUNCTION_ALPHA)
                ax.set_title(f't = {times_subset[frame]:.4f}')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                if plot_type == 'real':
                    ax.set_zlabel('Re(u)')
                elif plot_type == 'imag':
                    ax.set_zlabel('Im(u)')
                else:  # abs
                    ax.set_zlabel('|u|²')
                ax.set_zlim(vmin, vmax)
                ax.view_init(elev=30, azim=45)
                
                # Apply consistent layout for animation frames
                ax.set_box_aspect(None, zoom=0.85)
                
                # Save frame if requested
                if save_frames:
                    frame_path = os.path.join(frame_dir, f'frame_{frame:04d}.png')
                    plt.savefig(frame_path, dpi=300, bbox_inches='tight')
                
                return surf_num,
        else:
            # Initial scatter plots
            scatter_num = ax1.scatter(X[:, 0], X[:, 1], c=get_values(solutions_subset[0]),
                                    cmap=WAVEFUNCTION_2D_CMAP, vmin=vmin, vmax=vmax)
            plt.colorbar(scatter_num, ax=ax1)
            ax1.set_title(f't = {times_subset[0]:.4f}')
            
            if self.analytical_solution is not None:
                scatter_exact = ax2.scatter(X[:, 0], X[:, 1], c=get_values(None, True, 0),
                                          cmap=WAVEFUNCTION_2D_CMAP, vmin=vmin, vmax=vmax)
                plt.colorbar(scatter_exact, ax=ax2)
                ax2.set_title(f't = {times_subset[0]:.4f}')
            
            def update(frame):
                ax1.clear()
                values_num = get_values(solutions_subset[frame])
                scatter_num = ax1.scatter(X[:, 0], X[:, 1], c=values_num,
                                        cmap=WAVEFUNCTION_2D_CMAP, vmin=vmin, vmax=vmax)
                ax1.set_title(f't = {times_subset[frame]:.4f}')
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                
                if self.analytical_solution is not None:
                    ax2.clear()
                    values_exact = get_values(None, True, frame)
                    scatter_exact = ax2.scatter(X[:, 0], X[:, 1], c=values_exact,
                                              cmap=WAVEFUNCTION_2D_CMAP, vmin=vmin, vmax=vmax)
                    ax2.set_title(f't = {times_subset[frame]:.4f}')
                    ax2.set_xlabel('x')
                    ax2.set_ylabel('y')
                
                # Save frame if requested
                if save_frames:
                    frame_path = os.path.join(frame_dir, f'frame_{frame:04d}.png')
                    plt.savefig(frame_path, dpi=300, bbox_inches='tight')
                
                return scatter_num,
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                     interval=interval_ms, blit=True)
        
        # Save animation
        if save_path is None:
            save_path = f'figures/solution_animation_{plot_type}_{"3d" if plot_3d else "2d"}.gif'
        anim.save(save_path, writer='pillow', fps=int(fps))
        plt.close()
        
        if save_frames:
            print(f"Animation frames saved to: {frame_dir}")
        print(f"Animation saved to: {save_path} ({n_frames} frames, {fps:.1f} fps, {total_duration}s duration)")
    
    def plot_error_evolution(self, save_path: Optional[str] = None):
        """Plot the L2 error evolution over time."""
        if not self.errors:
            raise ValueError("No errors computed. Run solve() with compute_errors=True")
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(self.saved_times, self.errors, 'b-', linewidth=2, label='L2 Error')
        plt.xlabel('Time')
        plt.ylabel('L2 Error')
        plt.title('Error Evolution Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path is None:
            save_path = 'figures/error_evolution.png'
        
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Error evolution plot saved to: {save_path}")


def run_example():
    """Run example simulation and create visualizations."""
    print("\nRunning Schrödinger equation example...")

    class ModelPotentialClass:
        def __init__(self):
            self.func = ModelPotential(
                a=10000.0,
                b=100000.0,
                c=25000.0,
                make_asymmetric=True,
                time_dependent=True, # If False, ignore the rest of the parameters
                laser_amplitude=5000,
                laser_omega=3.0,
                laser_pulse_duration=0.4,
                laser_center_time=0.5,
                laser_envelope_type='gaussian',
                laser_spatial_profile_type='uniform',
                laser_charge=1.0,
                laser_polarization='linear_xy' # y: make only double well wiggle
            )
            self.t = 0.0

        def __call__(self, x: np.ndarray) -> np.ndarray:
            # Check if the underlying function is time-dependent
            if self.func.time_dependent:
                # Time-dependent case: call with (x, t)
                return self.func(x, self.t)
            else:
                # Static case: call with only x
                return self.func(x)

    # Create potential
    model_potential = ModelPotentialClass()   

    # Set time-independent potential
    model_potential.func.time_dependent = False

    # Create stationary solver
    stationary_solver = StationarySchrodingerSolver(
        nx=64, 
        ny=64,
        potential=model_potential
        )
    
    # Solve for multiple eigenvalues
    print("Solving for multiple eigenvalues...")
    max_energy = stationary_solver.estimate_energy_cutoff(safety_factor=0.2)
    eigenvalues, eigenfunctions = stationary_solver.solve_eigenvalues(
        n_eigenvalues=9, 
        max_energy=max_energy
    )
    
    # Plot eigenvalue spectrum
    stationary_solver.plot_eigenvalue_spectrum(save_path='figures/eigenvalue_spectrum.png', show=False)
    
    # Plot all eigenfunctions
    stationary_solver.plot_all_eigenfunctions(n_modes=9, save_path='figures/all_eigenfunctions.png', show=False)
    
    # Get ground state as initial condition for time-dependent solver
    initial_condition = stationary_solver.get_ground_state_as_initial_condition()
    print(f"Ground state eigenvalue: {stationary_solver.ground_state_eigenvalue:.6f}")

    # Set time-dependent potential
    model_potential.func.time_dependent = True

    # Plot the potential used in the simulation
    print("\nCreating potential visualizations...")
    model_potential.func.plot(
        n_points=100,
        plot_3d=True,
        save_path='figures/potential_animation_3d.gif',
        save_frames=False  # Set to True to save individual frames
    )
    
    # Create time-dependent solver, with the ground state as initial condition
    time_solver = SchrodingerSolver(
        nx=64, 
        ny=64,
        T_final=1.0, 
        dt=0.00001,
        potential=model_potential,
        initial_condition=initial_condition,  # Use filtered ground state
        analytical_solution=None
    )
    
    # Solve the equation
    results = time_solver.solve(store_solutions=True, compute_errors=True, save_interval=1000, save_frames=False)  # Set to True to save individual frames
    
    # Plot error evolution
    time_solver.plot_error_evolution(save_path='figures/error_evolution.png')
    
    # Plot initial state (ground state) in 2D and 3D
    print("\nPlotting initial state (ground state)...")
    time_solver.plot_solution(0.0, plot_type='abs', save_path='figures/initial_state_2d.png', show=False)
    time_solver.plot_solution(0.0, plot_type='abs', plot_3d=True, save_path='figures/initial_state_3d.png', show=False)
    
    # Plot final state in 2D and 3D
    time_solver.plot_solution(time_solver.T_final, plot_type='abs', save_path='figures/final_state_2d.png', show=False)
    time_solver.plot_solution(time_solver.T_final, plot_type='abs', plot_3d=True, save_path='figures/final_state_3d.png', show=False)
    
    # Create animations
    time_solver.animate_solution(plot_type='abs', plot_3d=False, save_path='figures/animation_2d.gif', save_frames=False)
    time_solver.animate_solution(plot_type='abs', plot_3d=True, save_path='figures/animation_3d.gif', save_frames=True)
    
    
    return time_solver


if __name__ == "__main__":
    # Run example
    time_solver = run_example()
    
