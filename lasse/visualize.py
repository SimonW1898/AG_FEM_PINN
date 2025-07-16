import numpy as np
import matplotlib.pyplot as plt
from potentials import ModelPotential
from schroedinger import SchrodingerSolver, StationarySchrodingerSolver
import os

def plot_potential():
    
    model_potential = ModelPotential(
        a=10000.0,
        b=100000.0,
        c=25000.0,
        make_asymmetric=True,
        time_dependent=True, # If False, ignore the rest of the parameters
        laser_amplitude=10000,
        laser_omega=3.0,
        laser_pulse_duration=0.4,
        laser_center_time=0.5,
        laser_envelope_type='gaussian',
        laser_spatial_profile_type='uniform',
        laser_charge=1.0,
        laser_polarization='linear_xy' # y: make only double well wiggle
    )

    print("\nCreating potential visualizations...")
    model_potential.plot(
        n_points=100,
        plot_3d=True,
        save_path='figures/potential_animation_3d.gif',
        save_frames=True  # Set to True to save individual frames
    )

def plot_l2_error_vs_time_different_resolutions():
    """
    Create L2 error vs time plot for different grid resolutions.
    Inspired by the error analysis in schroedinger.ipynb.
    """
    print("Creating L2 error vs time plot for different grid resolutions...")
    
    # Define grid resolutions to test
    grid_resolutions = [8, 16, 32, 64, 128, 256]  # Reduced for faster computation
    
    # Simulation parameters
    T_final = 1.0
    dt = 0.001  # Time step
    save_interval = 10  # Save every 10 time steps
    
    # Define analytical solution function
    def analytical_solution(x, t):
        """Analytical solution: u(x,y,t) = sin(πx) sin(πy) exp(-i 2π²t)"""
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-1j * 2 * np.pi**2 * t)
    
    # Store results for each grid resolution
    results = {}
    
    # Run simulations for each grid resolution
    for i, nx in enumerate(grid_resolutions):
        print(f"\nRunning simulation for nx=ny={nx} ({i+1}/{len(grid_resolutions)})")
        
        # Create time-dependent solver with analytical solution
        time_solver = SchrodingerSolver(
            nx=nx, 
            ny=nx,
            T_final=T_final, 
            dt=dt,
            potential=None,  # No potential (free Schrödinger equation)
            initial_condition=lambda x: analytical_solution(x, 0.0),  # Use analytical solution at t=0
            analytical_solution=analytical_solution  # Use analytical solution for error computation
        )
        
        # Solve the equation
        solver_results = time_solver.solve(
            store_solutions=True, 
            compute_errors=True,  # Compute actual L2 errors against analytical solution
            save_interval=save_interval, 
            save_frames=False, 
            save_solver=False
        )
        
        # Store results
        results[nx] = {
            'times': solver_results['times'],
            'solutions': solver_results['solutions'],
            'errors': solver_results['errors']
        }
        
        print(f"Completed nx=ny={nx}: {len(solver_results['times'])} time points")
        print(f"  Error range: {min(solver_results['errors']):.2e} to {max(solver_results['errors']):.2e}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Use different colors for each grid resolution
    colors = ['royalblue', 'crimson', 'darkorange', 'forestgreen', 'purple', 'brown']
    
    # Plot error evolution for each grid resolution
    for i, nx in enumerate(grid_resolutions):
        times = results[nx]['times']
        errors = results[nx]['errors']
        h = 1.0/nx  # Grid spacing
        
        plt.semilogy(times, errors, color=colors[i], linewidth=2, 
                    label=f'nx=ny={nx} (h={h:.3f})', marker='o', markersize=4)
    
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('L2 Error', fontsize=14)
    plt.title('L2 Error vs Time\n(Different Grid Resolutions)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Ensure y-axis shows the full range of errors
    all_errors = []
    for nx in grid_resolutions:
        all_errors.extend(results[nx]['errors'])
    
    y_min = min(all_errors)
    y_max = max(all_errors)
    plt.ylim(y_min * 0.5, y_max * 2.0)  # Add some padding
    
    print(f"Y-axis limits: {y_min:.2e} to {y_max:.2e}")
    
    plt.tight_layout()
    
    # Save the plot
    save_path = 'figures/l2_error_vs_time_different_resolutions.png'
    os.makedirs('figures', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"L2 error vs time plot saved to: {save_path}")
    
    return results

def plot_l2_error_vs_grid_resolution():
    """
    Create L2 error vs grid resolution plot for different time steps.
    """
    print("Creating L2 error vs grid resolution plot...")
    
    # Define grid resolutions to test
    grid_resolutions = [8, 16, 32, 64, 128, 256]
    
    # Simulation parameters
    T_final = 1.0
    dt = 0.01
    save_interval = 10
    
    # Define analytical solution function
    def analytical_solution(x, t):
        """Analytical solution: u(x,y,t) = sin(πx) sin(πy) exp(-i 2π²t)"""
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.exp(-1j * 2 * np.pi**2 * t)
    
    # Store results for each grid resolution
    results = {}
    
    # Run simulations for each grid resolution
    for i, nx in enumerate(grid_resolutions):
        print(f"\nRunning simulation for nx=ny={nx} ({i+1}/{len(grid_resolutions)})")
        
        # Create time-dependent solver with analytical solution
        time_solver = SchrodingerSolver(
            nx=nx, 
            ny=nx,
            T_final=T_final, 
            dt=dt,
            potential=None,  # No potential (free Schrödinger equation)
            initial_condition=lambda x: analytical_solution(x, 0.0),  # Use analytical solution at t=0
            analytical_solution=analytical_solution  # Use analytical solution for error computation
        )
        
        # Solve the equation
        solver_results = time_solver.solve(
            store_solutions=True, 
            compute_errors=True,  # Compute actual L2 errors against analytical solution
            save_interval=save_interval, 
            save_frames=False, 
            save_solver=False
        )
        
        # Store results
        results[nx] = {
            'times': solver_results['times'],
            'solutions': solver_results['solutions'],
            'errors': solver_results['errors']
        }
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Define time points to analyze
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = ['royalblue', 'crimson', 'darkorange', 'forestgreen', 'purple']
    
    # Plot error vs grid resolution for different times
    for i, t in enumerate(time_points):
        errors = []
        grid_spacings = []
        
        for nx in grid_resolutions:
            times = results[nx]['times']
            errors_list = results[nx]['errors']
            
            # Find closest time index
            time_idx = np.argmin(np.abs(np.array(times) - t))
            actual_time = times[time_idx]
            
            # Get error at this time
            error = errors_list[time_idx]
            errors.append(error)
            
            # Grid spacing
            grid_spacings.append(1.0/nx)
        
        plt.loglog(grid_spacings, errors, color=colors[i], linewidth=2, 
                  label=f't={t:.2f}', marker='o', markersize=6)
    
    plt.xlabel('Grid Spacing (h)', fontsize=14)
    plt.ylabel('L2 Error', fontsize=14)
    plt.title('L2 Error vs Grid Spacing\n(Different Times)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    save_path = 'figures/l2_error_vs_grid_resolution.png'
    os.makedirs('figures', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"L2 error vs grid resolution plot saved to: {save_path}")
    
    return results

if __name__ == "__main__":
    # Create potential visualization
    # plot_potential()
    
    # Create error analysis plots
    plot_l2_error_vs_time_different_resolutions()
    # plot_l2_error_vs_grid_resolution() 