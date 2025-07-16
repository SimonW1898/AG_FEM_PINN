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

    def check_convergence(self, other_solver):
        """
        Check eigenvalue convergence against another solver with different grid.
        
        Args:
            other_solver: Another StationarySchrodingerSolver with different grid
        """
        if self.eigenvalues is None or other_solver.eigenvalues is None:
            raise RuntimeError("Both solvers must have computed eigenvalues")
        
        n_compare = min(len(self.eigenvalues), len(other_solver.eigenvalues))
        
        print(f"Convergence check (first {n_compare} eigenvalues):")
        print(f"{'Index':<5} {'Coarse Grid':<12} {'Fine Grid':<12} {'Rel. Error':<12}")
        print("-" * 50)
        
        for i in range(n_compare):
            rel_error = abs(self.eigenvalues[i] - other_solver.eigenvalues[i]) / other_solver.eigenvalues[i]
            print(f"{i:<5} {self.eigenvalues[i]:<12.6f} {other_solver.eigenvalues[i]:<12.6f} {rel_error:<12.2e}")

    def analyze_spurious_modes(self, reference_eigenvalues=None):
        """
        Analyze which modes might be spurious by checking oscillation patterns.
        """
        if self.eigenfunctions is None:
            raise RuntimeError("Must solve eigenvalues first")
            
        print("\nMode Analysis:")
        print(f"{'Index':<5} {'Energy':<12} {'Nodes X':<8} {'Nodes Y':<8} {'Status':<10}")
        print("-" * 50)
        
        for i, (eig_val, phi) in enumerate(zip(self.eigenvalues, self.eigenfunctions)):
            # Count zero crossings (rough estimate of oscillations)
            values = phi.x.array.real
            X = self.V_space.tabulate_dof_coordinates()
            
            # Estimate number of nodes by checking sign changes
            # This is a rough heuristic - more sophisticated analysis possible
            x_coords = X[:, 0]
            y_coords = X[:, 1]
            
            # Count approximate nodes (this is simplified)
            x_nodes = "~"  # Placeholder - would need more sophisticated analysis
            y_nodes = "~"
            
            # Simple heuristic: very high energy modes are likely spurious
            h = 1.0 / max(self.nx, self.ny)
            max_reasonable_energy = (10 * np.pi / h) ** 2
            
            status = "OK" if eig_val < max_reasonable_energy else "Spurious?"
            
            print(f"{i:<5} {eig_val:<12.6f} {x_nodes:<8} {y_nodes:<8} {status:<10}")

    def plot_eigenfunction(self, mode_index=0, plot_type='abs', title=None, save_path=None, show=True):
        """Plot a specific eigenfunction."""
        if self.eigenfunctions is None:
            raise RuntimeError("Must solve eigenvalues first")
            
        if mode_index >= len(self.eigenfunctions):
            raise IndexError(f"Mode index {mode_index} out of range")
            
        phi = self.eigenfunctions[mode_index]
        eig_val = self.eigenvalues[mode_index]
        
        if title is None:
            title = f'Mode {mode_index}, E = {eig_val:.4f}'
        
        X = self.V_space.tabulate_dof_coordinates()
        if plot_type == 'real':
            Z = phi.x.array.real
            label = 'Re(ϕ)'
        elif plot_type == 'imag':
            Z = phi.x.array.imag
            label = 'Im(ϕ)'
        else:
            Z = np.abs(phi.x.array)**2
            label = '|ϕ|²'

        plt.figure(figsize=(8, 6))
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

# Usage example for double well potential
def create_double_well_potential(barrier_height=10.0, well_separation=0.5):
    """Create a double well potential in x-direction, constant in y."""
    def potential(x):
        x_vals = x[0]  # x-coordinates
        # Double well: V(x) = barrier_height * (x - 0.5)^4 - barrier_height * (x - 0.5)^2/4
        # Shifted to have wells at x ≈ 0.3 and x ≈ 0.7
        return barrier_height * ((x_vals - 0.5)**4 - (x_vals - 0.5)**2 / 4)
    return potential

# Example usage with convergence testing
def convergence_study():
    """Example of how to perform convergence study."""
    grids = [16, 32, 64]
    solvers = []
    
    # Create double well potential
    potential = create_double_well_potential(barrier_height=50.0)
    
    for nx in grids:
        print(f"\nSolving on {nx}x{nx} grid...")
        solver = StationarySchrodingerSolver(potential=potential, nx=nx, ny=nx)
        
        # Estimate energy cutoff
        max_energy = solver.estimate_energy_cutoff(safety_factor=0.2)
        print(f"Using energy cutoff: {max_energy:.2f}")
        
        # Solve for first 10 eigenvalues
        eigenvalues, eigenfunctions = solver.solve_eigenvalues(
            n_eigenvalues=10, 
            max_energy=max_energy
        )
        
        solver.analyze_spurious_modes()
        solvers.append(solver)
    
    # Check convergence between grids
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS")
    print("="*60)
    
    for i in range(len(solvers)-1):
        print(f"\nGrid {grids[i]} vs Grid {grids[i+1]}:")
        solvers[i].check_convergence(solvers[i+1])
    
    return solvers

if __name__ == "__main__":
    # Create solvers with different grid densities
    potential = create_double_well_potential(barrier_height=50.0)

    # Coarse grid
    solver_coarse = StationarySchrodingerSolver(potential=potential, nx=32, ny=32)
    eigenvals_coarse, eigenfuncs_coarse = solver_coarse.solve_eigenvalues(
        n_eigenvalues=6, 
        max_energy=solver_coarse.estimate_energy_cutoff(0.2)
    )

    # Fine grid  
    solver_fine = StationarySchrodingerSolver(potential=potential, nx=64, ny=64)
    eigenvals_fine, eigenfuncs_fine = solver_fine.solve_eigenvalues(
        n_eigenvalues=6,
        max_energy=solver_fine.estimate_energy_cutoff(0.2)
    )

    # Check convergence
    solver_coarse.check_convergence(solver_fine)

    # Plot first few modes
    for i in range(4):
        solver_fine.plot_eigenfunction(i, title=f'Mode {i}')