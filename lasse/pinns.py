import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import itertools
import time

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

def plot_3d_prediction(
    model=None, 
    checkpoint_path=None,
    t=0.0, 
    N=50, 
    component='abs', 
    save_path=None, 
    show=True, 
    analytical_solution=None, 
    ):
    """
    Create a 3D surface plot of PINN predictions over x-y grid at a given time.
    
    Args:
        model: Trained PINN model (required if checkpoint_path is provided, optional otherwise)
        checkpoint_path (str): Path to model checkpoint file (optional)
        t (float): Time value (0 to 1)
        N (int): Number of points in each spatial dimension
        component (str): 'real', 'imag', or 'abs' part of the solution
        save_path (str): Path to save the plot (optional)
        show (bool): Whether to display the plot
        analytical_solution (callable): Optional analytical solution function
        is_animation (bool): If True, don't use special initial solution color
    """
    # Load model state from checkpoint if provided
    if checkpoint_path is not None:
        if model is None:
            raise ValueError("Model must be provided when checkpoint_path is specified")
        
        # Load state dict into the provided model
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Loaded model state from checkpoint: {checkpoint_path}")
    elif model is None:
        raise ValueError("Either model or checkpoint_path must be provided")
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

def plot_3d_prediction_animation(
    model=None, 
    checkpoint_path=None,
    N=50, 
    component='real', 
    time_steps=None,                             
    save_dir='3d_plots', 
    total_duration=5.0, 
    save_path=None, 
    save_frames=False, 
    analytical_solution=None
    ):
    """
    Create multiple 3D plots at different time steps for animation.
    
    Args:
        model: Trained PINN model (required if checkpoint_path is provided, optional otherwise)
        checkpoint_path (str): Path to model checkpoint file (optional)
        N (int): Number of points in each spatial dimension
        component (str): 'real', 'imag', or 'abs' part of the solution
        time_steps (list): List of time values to plot (default: 10 evenly spaced from 0 to 1)
        save_dir (str): Directory to save plots
        total_duration (float): Total duration of the animation in seconds
        save_path (str): Optional path to save the animation
        save_frames (bool): If True, save individual frames
        analytical_solution (callable): Optional analytical solution function
    """
    # Load model state from checkpoint if provided
    if checkpoint_path is not None:
        if model is None:
            raise ValueError("Model must be provided when checkpoint_path is specified")
        
        # Load state dict into the provided model
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Loaded model state from checkpoint: {checkpoint_path}")
    elif model is None:
        raise ValueError("Either model or checkpoint_path must be provided")
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

class PINN(nn.Module):
    def __init__(self, n_hidden=64, n_layers=3, activation='tanh'):
        """
        Initialize the Physics-Informed Neural Network (PINN) model.

        Args:
            n_hidden (int): Number of hidden units in each hidden layer.
            n_layers (int): Number of hidden layers.
            activation (str): Activation function to use ('tanh', 'relu', 'sigmoid', 'gelu').
        """
        super().__init__()
        self.activation = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU()
        }[activation]

        layers = [nn.Linear(3, n_hidden)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(n_hidden, 2)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)
    
    def loss_physics(self, x, potential_function=None):
        """
        PDE loss function

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. Shape: (N, 3), where the last dimension is the time.
        potential_function : callable, optional
        """
        u = self(x) # shape: (N, 2)

        # split into real and imaginary parts
        u_real = u[:, 0] # shape: (N,)
        u_imag = u[:, 1] # shape: (N,)

        # Compute gradients with respect to all inputs at once
        # First derivatives
        du_real_dx = torch.autograd.grad(
            outputs=u_real.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0] # shape: (N, 3)
        
        du_imag_dx = torch.autograd.grad(
            outputs=u_imag.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0] # shape: (N, 3)

        # Extract time derivatives
        du_real_dt = du_real_dx[:, 2] # shape: (N,)
        du_imag_dt = du_imag_dx[:, 2] # shape: (N,)

        # Extract spatial derivatives
        du_real_dx1 = du_real_dx[:, 0] # shape: (N,)
        du_real_dx2 = du_real_dx[:, 1] # shape: (N,)
        du_imag_dx1 = du_imag_dx[:, 0] # shape: (N,)
        du_imag_dx2 = du_imag_dx[:, 1] # shape: (N,)

        # Second derivatives for Laplacian
        d2u_real_dx1 = torch.autograd.grad(
            outputs=du_real_dx1.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0][:, 0] # shape: (N,)
        
        d2u_real_dx2 = torch.autograd.grad(
            outputs=du_real_dx2.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0][:, 1] # shape: (N,)
        
        d2u_imag_dx1 = torch.autograd.grad(
            outputs=du_imag_dx1.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0][:, 0] # shape: (N,)
        
        d2u_imag_dx2 = torch.autograd.grad(
            outputs=du_imag_dx2.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0][:, 1] # shape: (N,)

        # Laplacian
        lap_u_real = d2u_real_dx1 + d2u_real_dx2
        lap_u_imag = d2u_imag_dx1 + d2u_imag_dx2

        # potential function
        if potential_function is None:
            V = torch.zeros_like(u_real)
        else:
            V = potential_function(x) # shape: (N,) or (N, 1)
            if V.dim() > 1:
                V = V.squeeze()

        # Residuals
        R_real = -du_imag_dt + lap_u_real + V * u_real
        R_imag = du_real_dt + lap_u_imag + V * u_imag

        # compute loss
        loss = torch.mean(R_real**2 + R_imag**2)

        return loss

    def loss_norm(self, x, C=1.0):
        """
        L² norm loss function
        
        Computes the L² norm of the model output and penalizes deviation from target norm C.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor. Shape: (N, 3), where the last dimension is the time.
        C : float, optional
            Target L² norm. Default is 1.0.
        
        Returns
        -------
        torch.Tensor
            Loss value penalizing deviation from target L² norm.
        """
        
        # get model output
        u = self(x)  # shape: (N, 2)
        
        # split into real and imaginary parts
        u_real = u[:, 0]  # shape: (N,)
        u_imag = u[:, 1]  # shape: (N,)
        
        # compute squared magnitude at each point
        magnitude_squared = u_real**2 + u_imag**2  # shape: (N,)
        
        # compute L² norm (discrete approximation)
        l2_norm = torch.sqrt(torch.mean(magnitude_squared))  # scalar
        
        # penalize deviation from target norm
        loss = (C - l2_norm)**2
        
        return loss

    def loss_boundary(self, x):
        """
        Boundary loss function
        
        Enforces the boundary condition u(x,t) = 0 on ∂Ω × (0,T].
        
        Parameters
        ----------
        x : torch.Tensor
            The boundary input tensor. Shape: (N, 3), where the last dimension is the time.
        
        Returns
        -------
        torch.Tensor
            Boundary loss value.
        """
        
        # get model output
        u = self(x)  # shape: (N, 2)
        
        # split into real and imaginary parts
        u_real = u[:, 0]  # shape: (N,)
        u_imag = u[:, 1]  # shape: (N,)
        
        # compute boundary loss: |u(x,t) - 0|² = |u(x,t)|² = u_real² + u_imag²
        loss = torch.mean(u_real**2 + u_imag**2)
        
        return loss

    def loss_initial(self, x, initial_condition):
        """
        Initial condition loss function
        
        Parameters
        ----------
        x : torch.Tensor
            The initial condition input tensor. Shape: (N, 3), where the last dimension is the time.
        initial_condition : callable
            Function that computes the initial condition.
        """
        u = self(x)

        u_real = u[:, 0]
        u_imag = u[:, 1]

        u_init_real = initial_condition(x)
        u_init_imag = torch.zeros_like(u_init_real)

        res_r = u_real - u_init_real
        res_i = u_imag - u_init_imag

        loss = torch.mean(res_r**2 + res_i**2)
        
        return loss

    def total_loss(self, x_b, x_i, x_pde, lambda_bc=1.0, lambda_ic=1.0, lambda_pde=1.0, lambda_norm=0.0):
        """
        Total loss function combining all loss components
        """
        loss_bc = self.loss_boundary(x_b)
        loss_ic = self.loss_initial(x_i, self.initial_condition)
        loss_pde = self.loss_physics(x_pde, self.potential_function)
        loss_norm_value = self.loss_norm(x_pde)

        # Compute weighted sum
        total_loss = lambda_bc * loss_bc + lambda_ic * loss_ic + lambda_pde * loss_pde + lambda_norm * loss_norm_value
        
        # Normalize by sum of weights (only if not all weights are zero)
        weight_sum = lambda_bc + lambda_ic + lambda_pde + lambda_norm
        if weight_sum > 0:
            total_loss = total_loss / weight_sum
            
        return total_loss

    @staticmethod
    def sample_boundary_points(num_points, seed=42):
        """
        Sample points on the boundary of the domain
        """
        np.random.seed(seed)
        x1 = np.random.uniform(0, 1, num_points)
        x2 = np.random.uniform(0, 1, num_points)
        t = np.random.uniform(0, 1, num_points)
        # Round x1 and x2 to 0 or 1
        x1_rounded = np.round(x1)
        x2_rounded = np.round(x2)
        x_b = np.column_stack((x1_rounded, x2_rounded, t))
        return torch.tensor(x_b, dtype=torch.float32, requires_grad=True)

    @staticmethod
    def sample_initial_points(num_points, seed=42):
        """
        Sample points for initial condition (t=0)
        """
        np.random.seed(seed)
        x1 = np.random.uniform(0, 1, num_points)
        x2 = np.random.uniform(0, 1, num_points)
        t = np.zeros(num_points)
        x_i = np.column_stack((x1, x2, t))
        return torch.tensor(x_i, dtype=torch.float32, requires_grad=True)

    @staticmethod
    def sample_interior_points(num_points, seed=42):
        """
        Sample interior points for PDE loss
        """
        np.random.seed(seed)
        x1 = np.random.uniform(0, 1, num_points)
        x2 = np.random.uniform(0, 1, num_points)
        t = np.random.uniform(0, 1, num_points)
        x_i = np.column_stack((x1, x2, t))
        return torch.tensor(x_i, dtype=torch.float32, requires_grad=True)

    def potential_function(self, x):
        """
        Define the potential function V(x,t).
        For example, V(x,t) = 0.1 * (x1^2 + x2^2) * (1 + t)
        """
        x1 = x[:, 0]
        x2 = x[:, 1]
        t = x[:, 2]
        return 0.0 * (x1**2 + x2**2) * (1 + t) # set to zero for now, can be modified later

    def initial_condition(self, x):
        """
        Define the initial condition u(x,0)
        """
        x1 = x[:, 0]
        x2 = x[:, 1]
        return torch.sin(np.pi * x1) * torch.sin(np.pi * x2)

    def train_model(self, n_hidden=128, n_layers=3, lr=1e-4,
                    lambda_bc=1.0, lambda_ic=1.0, lambda_pde=1.0, lambda_norm=1.0,
                    n_epochs=500, n_points=50000, batch_size=2024,
                    activation='tanh', verbose=True):
        """
        Train the PINN model
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.to(device)

        # Define optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Load sample points
        x_b = self.sample_boundary_points(n_points).to(device).requires_grad_()
        x_i = self.sample_initial_points(n_points).to(device).requires_grad_()
        x_pde = self.sample_interior_points(n_points).to(device).requires_grad_()

        # Validation points
        n_val_points = int(n_points/10)
        x_val_b = self.sample_boundary_points(n_val_points).to(device).requires_grad_()
        x_val_i = self.sample_initial_points(n_val_points).to(device).requires_grad_()
        x_val_pde = self.sample_interior_points(n_val_points).to(device).requires_grad_()

        # Create DataLoaders
        boundary_loader = DataLoader(TensorDataset(x_b), batch_size=batch_size, shuffle=True)
        initial_loader = DataLoader(TensorDataset(x_i), batch_size=batch_size, shuffle=True)
        pde_loader = DataLoader(TensorDataset(x_pde), batch_size=batch_size, shuffle=True)

        # Training loop
        num_batches = max(len(boundary_loader), len(initial_loader), len(pde_loader))
        for epoch in range(n_epochs):
            self.train()
            epoch_loss = 0.0
            batch_counter = 0
            for (xb_batch,), (xi_batch,), (xpde_batch,) in zip(
                itertools.cycle(boundary_loader),
                itertools.cycle(initial_loader),
                pde_loader
            ):
                optimizer.zero_grad()
                loss = self.total_loss(xb_batch, xi_batch, xpde_batch, lambda_bc, lambda_ic, lambda_pde, lambda_norm)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_counter += 1
                if batch_counter >= num_batches:
                    break
            epoch_loss /= batch_counter

            if verbose and epoch % 100 == 0:
                self.eval()
                val_loss = self.total_loss(x_val_b, x_val_i, x_val_pde, lambda_bc, lambda_ic, lambda_pde, lambda_norm)
                print(f"Epoch {epoch:4d} | Training Loss: {epoch_loss:.4e} | Validation Loss: {val_loss.item():.4e}")
                print("Training Loss of one batch:")
                print(f"Boundary Loss: {self.loss_boundary(xb_batch).item():.4e}, Initial Loss: {self.loss_initial(xi_batch, self.initial_condition).item():.4e}, PDE Loss: {self.loss_physics(xpde_batch, self.potential_function).item():.4e}, Norm Loss: {self.loss_norm(xpde_batch).item():.4e}")
                print("Validation Losses:")
                print(f"Boundary Loss: {self.loss_boundary(x_val_b).item():.4e}, Initial Loss: {self.loss_initial(x_val_i, self.initial_condition).item():.4e}, PDE Loss: {self.loss_physics(x_val_pde, self.potential_function).item():.4e}, Norm Loss: {self.loss_norm(x_val_pde).item():.4e}\n")
                torch.save(self.state_dict(), f"pinn_model_in_training_{epoch:04d}.pth")

        # Final evaluation
        self.eval()
        final_val_loss = self.total_loss(x_val_b, x_val_i, x_val_pde, lambda_bc, lambda_ic, lambda_pde, lambda_norm)
        print(f"\nFinal Validation Loss: {final_val_loss.item():.4e}")
        # Save the model
        torch.save(self.state_dict(), f"pinn_model_{n_hidden}_{n_layers}_{lr:.0e}_{lambda_bc}_{lambda_ic}_{lambda_pde}.pth")
        print(f"Model saved as pinn_model_{n_hidden}_{n_layers}_{lr:.0e}_{lambda_bc}_{lambda_ic}_{lambda_pde}.pth") 

        return self


# Plotting functions (standalone)
def plot_loss_history(loss_history, epoch_history, window=500):
    """
    Plot the loss history during training
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_history, loss_history, label=f"Loss History")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.grid(True)
    plt.legend()
    plt.savefig("loss_history.png")
    plt.close()


if __name__ == "__main__":
    # choose hyperparameters
    #{'n_hidden': 64, 'n_layers': 3, 'lr': 3.4200466989175795e-05, 'lambda_bc': 0.1, 'lambda_ic': 0.1, 'lambda_pde': 1.0}
    n_hidden = 128
    n_layers = 3
    lr = 3.4200466989175795e-05
    lambda_bc = 1.0
    lambda_ic = 1.0
    lambda_pde = 1.0
    lambda_norm = 1.0 

    # other parameters
    n_epochs = 500
    n_points = 50000
    batch_size = 2024
    start = time.time()
    pinn_model = PINN()
    pinn_model.train_model(n_hidden=n_hidden, n_layers=n_layers, lr=lr,
                lambda_bc=lambda_bc, lambda_ic=lambda_ic, lambda_pde=lambda_pde, lambda_norm=lambda_norm,
                n_epochs=n_epochs, n_points=n_points, batch_size=batch_size)
    end = time.time()
    print(f"Training time: {(end - start)/60:.2f} minutes")