import torch
from PINN_tuning import PINN
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List
import os

# Global plotting configuration (similar to schroedinger.py)
WAVEFUNCTION_COLOR = 'crimson'
WAVEFUNCTION_ALPHA = 0.9
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

n_hidden = 256
n_epochs = 5000

# Load the model
model_name = f"../pinn_model_128_3_3e-05_01_01_10.pth"

n_hidden = 128
n_layers = 3
lr = 3.4200466989175795e-05
lambda_bc = 0.1
lambda_ic = 0.1
lambda_pde = 1.0
activation = 'tanh'

model = PINN(n_hidden=n_hidden, n_layers=n_layers, activation=activation)

model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
model.eval()
print(f"Loading model from {model_name}")

def get_pinn_prediction(model, x_coords, y_coords, t):
    """
    Get PINN predictions for given coordinates and time.
    
    Args:
        model: Trained PINN model
        x_coords: X coordinates (1D array)
        y_coords: Y coordinates (1D array) 
        t: Time value (scalar)
    
    Returns:
        predictions: Array of shape (n_points, 2) with [real, imaginary] parts
    """
    # Create grid
    X, Y = np.meshgrid(x_coords, y_coords)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    T_array = np.full(X_flat.shape, t)
    
    # Create input tensor and get predictions
    input_tensor = torch.tensor(np.column_stack((X_flat, Y_flat, T_array)), 
                               dtype=torch.float32, requires_grad=False)
    
    with torch.no_grad():
        predictions = model(input_tensor).numpy()
    
    return predictions, X, Y, X_flat, Y_flat

def plot_3d_prediction(model, t=0.0, N=50, component='abs', save_path=None, show=True, 
                      analytical_solution=None, is_animation=False):
    """
    Create a 3D surface plot of PINN predictions over x-y grid at a given time.
    
    Args:
        model: Trained PINN model
        t (float): Time value (0 to 1)
        N (int): Number of points in each spatial dimension
        component (str): 'real', 'imag', or 'abs' part of the solution
        save_path (str): Path to save the plot (optional)
        show (bool): Whether to display the plot
        analytical_solution (callable): Optional analytical solution function
        is_animation (bool): If True, don't use special initial solution color
    """
    # Create grid
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    
    # Get PINN predictions
    predictions, X, Y, X_flat, Y_flat = get_pinn_prediction(model, x, y, t)
    
    # Select component (real, imaginary, or absolute)
    if component.lower() == 'real':
        Z = predictions[:, 0].reshape(X.shape)
        title_component = 'Real'
        zlabel = 'Re(u)'
    elif component.lower() == 'imag':
        Z = predictions[:, 1].reshape(X.shape)
        title_component = 'Imaginary'
        zlabel = 'Im(u)'
    elif component.lower() == 'abs':
        Z = np.sqrt(np.sum(predictions**2.0, axis=1)).reshape(X.shape)
        title_component = 'Absolute Value'
        zlabel = '|u|²'
    else:
        raise ValueError("component must be 'real', 'imag', or 'abs'")
    
    # Calculate z-limits for consistency across time steps
    if not hasattr(plot_3d_prediction, '_plot_zlimits') or plot_3d_prediction._plot_zlimits is None:
        # Use initial condition (t=0) to set consistent limits
        initial_predictions, _, _, _, _ = get_pinn_prediction(model, x, y, 0.0)
        if component.lower() == 'real':
            first_values = initial_predictions[:, 0].reshape(X.shape)
        elif component.lower() == 'imag':
            first_values = initial_predictions[:, 1].reshape(X.shape)
        else:  # abs
            first_values = np.sqrt(np.sum(initial_predictions**2.0, axis=1)).reshape(X.shape)
        
        # Add analytical solution if available
        if analytical_solution is not None:
            # This would need to be implemented based on your analytical solution
            # For now, just use PINN values
            pass
        
        zmin = first_values.min()
        zmax = first_values.max()
        
        # Add some padding (handle edge cases)
        z_range = zmax - zmin
        if z_range < 1e-10:  # Very small range
            padding = 0.1 * max(abs(zmin), abs(zmax), 0.1)
        else:
            padding = 0.1 * z_range
        plot_3d_prediction._plot_zlimits = (zmin - padding, zmax + padding)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use initial solution color for static plots at t=0.0, otherwise use regular wavefunction color
    numerical_color = WAVEFUNCTION_COLOR
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, color=numerical_color, linewidth=0, alpha=WAVEFUNCTION_ALPHA)
    
    # Plot analytical solution if available
    if analytical_solution is not None:
        # This would need to be implemented based on your analytical solution
        # For now, just plot PINN solution
        pass
    
    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(zlabel)
    ax.set_title(f't = {t:.4f}')
    ax.set_zlim(plot_3d_prediction._plot_zlimits)
    
    # Set view angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Adjust layout for 3D plots
    ax.set_box_aspect(None, zoom=0.85)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_3d_prediction_animation(model, N=50, component='real', time_steps=None, 
                                save_dir='3d_plots', total_duration=5.0, save_path=None, 
                                save_frames=False, analytical_solution=None):
    """
    Create multiple 3D plots at different time steps for animation.
    
    Args:
        model: Trained PINN model
        N (int): Number of points in each spatial dimension
        component (str): 'real', 'imag', or 'abs' part of the solution
        time_steps (list): List of time values to plot (default: 10 evenly spaced from 0 to 1)
        save_dir (str): Directory to save plots
        total_duration (float): Total duration of the animation in seconds
        save_path (str): Optional path to save the animation
        save_frames (bool): If True, save individual frames
        analytical_solution (callable): Optional analytical solution function
    """
    import matplotlib.animation as animation
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    if time_steps is None:
        time_steps = np.linspace(0, 1, 10)
    
    # Calculate frame rate to achieve desired total duration
    n_frames = len(time_steps)
    fps = n_frames / total_duration
    interval_ms = 1000 / fps  # Convert to milliseconds for matplotlib
    
    # Create grid
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    
    # Get value range for consistent z-axis limits
    initial_predictions, _, _, _, _ = get_pinn_prediction(model, x, y, 0.0)
    if component.lower() == 'real':
        first_values = initial_predictions[:, 0].reshape((N, N))
    elif component.lower() == 'imag':
        first_values = initial_predictions[:, 1].reshape((N, N))
    else:  # abs
        first_values = np.sqrt(np.sum(initial_predictions**2.0, axis=1)).reshape((N, N))
    
    vmin = first_values.min()
    vmax = first_values.max()
    
    # Add some padding
    padding = 0.1 * (vmax - vmin)
    vmin -= padding
    vmax += padding
    
    # Create figure and axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set initial labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if component.lower() == 'real':
        ax.set_zlabel('Re(u)')
    elif component.lower() == 'imag':
        ax.set_zlabel('Im(u)')
    else:  # abs
        ax.set_zlabel('|u|²')
    ax.set_zlim(vmin, vmax)
    ax.view_init(elev=30, azim=45)
    
    def update(frame):
        ax.clear()
        t = time_steps[frame]
        
        # Get predictions for current time
        predictions, X, Y, _, _ = get_pinn_prediction(model, x, y, t)
        
        # Select component
        if component.lower() == 'real':
            Z = predictions[:, 0].reshape(X.shape)
        elif component.lower() == 'imag':
            Z = predictions[:, 1].reshape(X.shape)
        else:  # abs
            Z = np.sqrt(np.sum(predictions**2.0, axis=1)).reshape(X.shape)
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, color=WAVEFUNCTION_COLOR, linewidth=0, alpha=WAVEFUNCTION_ALPHA)
        
        # Set labels and title
        ax.set_title(f't = {t:.4f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if component.lower() == 'real':
            ax.set_zlabel('Re(u)')
        elif component.lower() == 'imag':
            ax.set_zlabel('Im(u)')
        else:  # abs
            ax.set_zlabel('|u|²')
        ax.set_zlim(vmin, vmax)
        ax.view_init(elev=30, azim=45)
        
        # Apply consistent layout for animation frames
        ax.set_box_aspect(None, zoom=0.85)
        
        # Save frame if requested
        if save_frames:
            frame_path = os.path.join(save_dir, f'pinn_frame_{frame:04d}.png')
            plt.savefig(frame_path, dpi=300, bbox_inches='tight')
        
        return surf,
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                 interval=interval_ms, blit=True)
    
    # Save animation
    if save_path is None:
        save_path = os.path.join(save_dir, f'pinn_animation_{component}_3d.gif')
    anim.save(save_path, writer='pillow', fps=int(fps))
    plt.close()
    
    if save_frames:
        print(f"Animation frames saved to: {save_dir}")
    print(f"Animation saved to: {save_path} ({n_frames} frames, {fps:.1f} fps, {total_duration}s duration)")

# Example usage:
if __name__ == "__main__":
    # Plot single 3D surface
    plot_3d_prediction(model, t=0.0, N=50, component='abs', 
                      save_path='3d_prediction_abs.png')
    
    # Create animation sequence
    plot_3d_prediction_animation(model, N=50, component='abs', 
                                time_steps=np.linspace(0, 1, 51), save_frames=True)