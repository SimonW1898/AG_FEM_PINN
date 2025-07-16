import numpy as np
import matplotlib.pyplot as plt
import torch
from problem_1_PINN import PINN, sample_boundary_points, sample_initial_points, sample_interior_points, plot_model_output, total_loss, load_sample_points

n_hidden = 256
n_epochs = 5000

# Load the model
model_name = f"pinn_model_{n_hidden}_{n_epochs}.pt"
model = PINN(n_hidden=n_hidden)
model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
model.eval()
print(f"Loading model from {model_name}")

# Define analytical solution
def analytical_solution(x, y, t):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-2 * np.pi**2 * t)

T = 1
nt = 50
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    error = np.linalg.norm(model_output[:, 0] - analytical_output)

    # Reshape error to grid
    error_grid = error.reshape(n_points, n_points)

    if i % 10 == 0:  # Save every 10th timestep
        # Plot error
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, error_grid, levels=50, cmap='viridis')
        plt.colorbar(contour, label='Error')
        plt.title(f'Error between PINN output and analytical solution at t={t:.4f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plot_path = f"simon/PINN_error/pinn_error_t{t:.4f}.png"
        plt.savefig(plot_path)
        plt.close()  # Close plot to save memory
    
    # Compute and print L2 error
    L2_error = np.sqrt(np.mean(error**2))
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