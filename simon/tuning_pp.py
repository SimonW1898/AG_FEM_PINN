import numpy as np
import matplotlib.pyplot as plt
import torch
from PINN_tuning import PINN, sample_boundary_points, sample_initial_points, sample_interior_points, plot_model_output, total_loss, load_sample_points, loss_physics


def sample_interior_points_at_T(n_points, T):
    """Sample interior points in the domain."""
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)
    t = T*np.ones(n_points)
    return torch.tensor(np.column_stack((x, y, t)), dtype=torch.float32).requires_grad_()



# Define analytical solution
def analytical_solution(x, y, t):
    """ 
    Return the analytical solution u(x, y, t) = sin(pi x) sin(pi y) exp(-i 2 pi^2 t)

    Input:
        x: x-coordinates (ndim)
        y: y-coordinates (ndim)
        t: time (ndim)
    Output:
        analytical solution (ndim x 2) # real and imaginary parts
    """
    spatial = np.sin(np.pi * x) * np.sin(np.pi * y)
    real_t = np.cos(2 * np.pi**2 * t)
    imag_t = -np.sin(2 * np.pi**2 * t)

    return np.column_stack([spatial * real_t, spatial * imag_t])

n_hidden = 256
n_epochs = 5000

# Load the model
# model_name = f"pinn_model_128_3_3e-05_01_01_10.pth"

# n_hidden = 128
# n_layers = 3
# lr = 3.4200466989175795e-05
# lambda_bc = 0.1
# lambda_ic = 0.1
# lambda_pde = 1.0
# activation = 'tanh'

model_name = f"/workspaces/new_repo_2/pinn_model_128_3_3e-05_1_1_1.pth"
n_hidden = 128
n_layers = 3
lr = 3.4200466989175795e-05
lambda_bc = 1.0
lambda_ic = 1.0
lambda_pde = 1.0
activation = 'tanh'


device = torch.device("cpu")
model = PINN(n_hidden=n_hidden, n_layers=n_layers, activation=activation)

model.load_state_dict(torch.load(model_name, map_location=device))
model.eval()
print(f"Loading model from {model_name}")


T = 1
nt = 100
time_steps = np.linspace(0, T, nt + 1)
error_timesteps = np.zeros((nt+1,1))
# Grid setup for evaluation
n_points = 100
x = np.linspace(0, 1, n_points)
y = np.linspace(0, 1, n_points)
X, Y = np.meshgrid(x, y)
X_flat = X.flatten()
Y_flat = Y.flatten()



n_sample = 50000
# Load sample points
x_b, x_i, x_pde = load_sample_points(f"sample_points_{n_sample}.npz")
# get device

# to device to compute the loss
x_b = x_b.to(device)
x_i = x_i.to(device)
x_pde = x_pde.to(device)


for i,t in enumerate(time_steps):
    # Prepare input tensor with current time
    T_array = np.full(X_flat.shape, t)
    X_tensor = torch.tensor(np.column_stack((X_flat, Y_flat, T_array)), dtype=torch.float32, requires_grad=True)
    
    # Evaluate model output
    model_output = model(X_tensor).detach().numpy()
    
    # Calculate analytical solution
    analytical_output = analytical_solution(X_flat, Y_flat, t)
    
    # Calculate error
    error = (model_output - analytical_output)


    # Reshape error to grid
    error_grid = error.reshape(n_points, n_points, 2)

    
    if i % 10 == 0:  # Save every 10th timestep
        # Plot error
        plt.subplot(1, 2, 1)
        contour = plt.contourf(X, Y, error_grid[:, :, 0], levels=50, cmap='viridis')
        plt.colorbar(contour, label='Error (Real Part)')
        plt.title(f'real at t={t:.4f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.subplot(1, 2, 2)
        contour = plt.contourf(X, Y, error_grid[:, :, 1], levels=50, cmap='viridis')
        plt.colorbar(contour, label='Error (Imaginary Part)')
        plt.title(f'imag at t={t:.4f}')
        plt.xlabel('x')
        plt.ylabel('y')

        plot_path = f"simon/PINN_error/pinn_error_t{t:.4f}.png"
        plt.savefig(plot_path)
        plt.close()  # Close plot to save memory
    
    # Compute and print L2 error
    L2_error = np.sqrt(np.mean(np.square(error[:, 0]) + np.square(error[:, 1])))
    error_timesteps[i] = L2_error
    print(f"L2 error at t={t:.4f}: {L2_error:.6f}")
    # print(f"Loss at t={t:.4f}: {total_loss(model, x_b, x_i, x_pde):.6f}")

# plot and store plot
plt.figure(figsize=(10, 6))
plt.plot(time_steps, error_timesteps, marker='o', label='L2 Error')
plt.xlabel('Time (t)')
plt.ylabel('L2 Error')
plt.title('L2 Error over Time')
plt.grid(True)
plt.legend()
plt.savefig("simon/PINN_error/L2_error_over_time.png")
plt.close()



# compute pde loss
for i,t in enumerate(time_steps):
    X = sample_interior_points_at_T(5000, t)
    X_tensor = X.to(device).requires_grad_()
    # check devices for X_tensor and model
    pde_loss = loss_physics(model, X_tensor)
    print(f"PDE Loss at t={t:.4f}: {pde_loss.item():.6f}")