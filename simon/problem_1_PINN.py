import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import itertools

import numpy as np
import matplotlib.pyplot as plt
import time

# optuna check out for hyperparameter optimization

# PINN model definition
class PINN(nn.Module):
    def __init__(self, n_hidden = 64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(3, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, n_hidden)
        self.linear4 = nn.Linear(n_hidden, 2)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        output = self.linear4(x)
        return output

# Derivative utilities
def get_gradients(model, x):
    output = model(x)
    grads = []
    for i in range(output.shape[1]):
        grad = torch.autograd.grad(
            outputs=output[:, i],
            inputs=x,
            grad_outputs=torch.ones_like(output[:, i]),
            create_graph=True,
            retain_graph=True
        )[0]
        grads.append(grad)
    gradients = torch.stack(grads, dim=1)
    return gradients

def second_derivative(y, x, idx):
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0][:, idx]
    return grad

# Loss functions
def loss_physics(model, x):
    """ 
    minimize residual of the PDE
    \mathcal{R}(x,t) := i \, \partial_t u_\theta(x,t) + \Delta u_\theta(x,t) - V(x,t) u_\theta(x,t).
    """


    u = model(x)
    # first derivatives
    grads = get_gradients(model, x)
    dx1 = grads[:, :, 0]
    dx2 = grads[:, :, 1]
    dt = grads[:, :, 2]

    # real part derivatives
    u_r_x1 = dx1[:, 0]
    u_r_x2 = dx2[:, 0]
    # imaginary part derivatives
    u_i_x1 = dx1[:, 1]
    u_i_x2 = dx2[:, 1]

    # real second derivatives
    u_r_x1x1 = second_derivative(u_r_x1, x, 0)
    u_r_x2x2 = second_derivative(u_r_x2, x, 1)
    # imaginary second derivatives
    u_i_x1x1 = second_derivative(u_i_x1, x, 0)
    u_i_x2x2 = second_derivative(u_i_x2, x, 1)


    # Laplacian of u_r
    lap_u_r = u_r_x1x1 + u_r_x2x2
    # Laplacian of u_i
    lap_u_i = u_i_x1x1 + u_i_x2x2

    # time derivative
    u_r_t = dt[:, 0]
    u_i_t = dt[:, 1]

    # potential term needs to be added
    v_u_r = potential_function(x) * u[:, 0]  # real part
    v_u_i = potential_function(x) * u[:, 1]  # imaginary part

    # residuals real and imaginary parts
    res_real = -u_i_t + lap_u_r + v_u_r             # -u_i_t comes from i* u which turns the imaginary part of u into real with - sign
    res_imag = u_r_t + lap_u_i + v_u_i

    loss = torch.mean(res_real**2 + res_imag**2)
    return loss

def loss_nontrivial(model, x):
    """
    Non-trivial loss function to ensure the model is not just learning the trivial solution.
    This can be a simple regularization term or a more complex constraint.
    """
    u = model(x)
    u_r = u[:, 0]
    u_i = u[:, 1]

    loss = (0.5 - torch.mean(u_r**2 + u_i**2))**2  # encourage non-trivial solutions
    return loss

def potential_function(x):
    """
    Define the potential function V(x,t).
    For example, V(x,t) = 0.1 * (x1^2 + x2^2) * (1 + t)
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    t = x[:, 2]
    return 0.0 * (x1**2 + x2**2) * (1 + t) # set to zero for now, can be modified later

def loss_boundary(model, x_b):
    u_b = model(x_b)
    loss = torch.mean(u_b[:, 0]**2 + u_b[:, 1]**2)
    return loss

def loss_initial(model, x_i):
    u_i = model(x_i)
    initial_r = torch.sin(np.pi * x_i[:, 0]) * torch.sin(np.pi * x_i[:, 1])
    initial_i = torch.zeros_like(initial_r)
    res_r = u_i[:, 0] - initial_r
    res_i = u_i[:, 1] - initial_i
    loss = torch.mean(res_r**2 + res_i**2)
    return loss

def total_loss(model, x_b, x_i, x_pde, lambda_bc=1.0, lambda_ic=1.0, lambda_pde=1.0, lambda_nontrivial=0.0):
    loss_bc = loss_boundary(model, x_b)
    loss_ic = loss_initial(model, x_i)
    loss_pde = loss_physics(model, x_pde)
    loss_nontrivial_value = loss_nontrivial(model, x_pde)

    sum_lambda = lambda_bc + lambda_ic + lambda_pde + lambda_nontrivial
    total_loss = (lambda_bc * loss_bc + lambda_ic * loss_ic + lambda_pde * loss_pde + lambda_nontrivial * loss_nontrivial_value) / sum_lambda
    return total_loss


def sample_boundary_points(num_points, T=0.1):
    x1 = np.zeros(num_points)
    x2 = np.zeros(num_points)
    t = np.random.uniform(0, T, num_points)
    num_per_edge = num_points // 4
    x1[:num_per_edge] = 0.0
    x2[:num_per_edge] = np.random.uniform(0, 1, num_per_edge)
    x1[num_per_edge:2*num_per_edge] = 1.0
    x2[num_per_edge:2*num_per_edge] = np.random.uniform(0, 1, num_per_edge)
    x1[2*num_per_edge:3*num_per_edge] = np.random.uniform(0, 1, num_per_edge)
    x2[2*num_per_edge:3*num_per_edge] = 0.0
    x1[3*num_per_edge:] = np.random.uniform(0, 1, num_points - 3*num_per_edge)
    x2[3*num_per_edge:] = 1.0
    x_b = np.column_stack((x1, x2, t))
    return torch.tensor(x_b, dtype=torch.float32, requires_grad=True)

def sample_initial_points(num_points):
    x1 = np.random.uniform(0, 1, num_points)
    x2 = np.random.uniform(0, 1, num_points)
    t = np.zeros(num_points)
    x_i = np.column_stack((x1, x2, t))
    return torch.tensor(x_i, dtype=torch.float32, requires_grad=True)

def sample_interior_points(num_points, T=0.1):
    x1 = np.random.uniform(0, 1, num_points)
    x2 = np.random.uniform(0, 1, num_points)
    t = np.random.uniform(0, T, num_points)
    x_i = np.column_stack((x1, x2, t))
    return torch.tensor(x_i, dtype=torch.float32, requires_grad=True)



#  Plotting function
def plot_model_output(model, t=0):
    x1 = np.linspace(0, 1, 100)
    x2 = np.linspace(0, 1, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.column_stack((X1.ravel(), X2.ravel(), np.full(X1.size, t)))
    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    X_tensor = X_tensor.to(next(model.parameters()).device)  # move input to model's device
    u_pred = model(X_tensor).detach().cpu().numpy()          # detach and move result back to CPU for plotting

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    cf1 = axes[0].contourf(X1, X2, u_pred[:, 0].reshape(X1.shape), levels=50, cmap='viridis')
    fig.colorbar(cf1, ax=axes[0], label='Real Part of u')
    axes[0].set_title(f'Real Part of the Solution at t={t}')
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')
    cf2 = axes[1].contourf(X1, X2, u_pred[:, 1].reshape(X1.shape), levels=50, cmap='viridis')
    fig.colorbar(cf2, ax=axes[1], label='Imaginary Part of u')
    axes[1].set_title(f'Imaginary Part of the Solution at t={t}')
    axes[1].set_xlabel('x1')
    axes[1].set_ylabel('x2')
    plt.tight_layout()
    plt.savefig(f"model_output_t{t}.png")
    plt.close()

def plot_loss_history(loss_history, epoch_history, window=500):
    # def moving_average(x, w):
    #     return np.convolve(x, np.ones(w)/w, mode='valid')
    
    # smoothed_loss = moving_average(loss_history, window)
    
    # smoothed_epochs = epoch_history[window-1:]
    

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_history, loss_history, label=f"Smoothed (window={window})")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Smoothed Loss History')
    plt.grid(True)
    plt.legend()
    plt.savefig("loss_history.png")
    plt.close()


def make_sample_points_file_name(n_points):
    N_points_interior = int(n_points)
    N_points_initial = int(n_points)
    N_points_boundary = int(n_points)
    x_b = sample_boundary_points(N_points_boundary, T)
    x_i = sample_initial_points(N_points_initial)
    x_pde = sample_interior_points(N_points_interior, T)
        
    # store sample points in file

    np.savez(f"sample_points_{N_points_interior}.npz", x_b=x_b.cpu().detach().numpy(), x_i=x_i.cpu().detach().numpy(), x_pde=x_pde.cpu().detach().numpy())


def load_sample_points(file_path):
    data = np.load(file_path)
    x_b = torch.tensor(data['x_b'], dtype=torch.float32, requires_grad=True)
    x_i = torch.tensor(data['x_i'], dtype=torch.float32, requires_grad=True)
    x_pde = torch.tensor(data['x_pde'], dtype=torch.float32, requires_grad=True)
    return x_b, x_i, x_pde




if __name__ == "__main__":
    start = time.time()
    # check cuda availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")



    # Sampling functions
    T = 1

    # Training
    n_hidden = 64
    model = PINN(n_hidden=n_hidden)
    # shift model on device
    model.to(device)

    n_points = int(50000)
    print("Creating sample points...")
    # make sample points file name
    make_sample_points_file_name(n_points)
    # Load sample points
    x_b, x_i, x_pde = load_sample_points(f"sample_points_{n_points}.npz")
    print("Sample points loaded.")
    

    # Move training data to device
    x_b = x_b.to(device)
    x_i = x_i.to(device)
    x_pde = x_pde.to(device)
    
    


    optimizer = optim.Adam(
        params=model.parameters(),
        lr=0.0005,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False
    )


    epochs = 2000
    patience = 10000
    best_loss = float('inf')
    training_state = 0
    loss_history = []
    epoch_history = []
    # Create TensorDatasets
    batch_size = 1024  # You can adjust this
    boundary_dataset = TensorDataset(x_b)
    initial_dataset = TensorDataset(x_i)
    pde_dataset = TensorDataset(x_pde)

    # Create DataLoaders
    boundary_loader = DataLoader(boundary_dataset, batch_size=batch_size, shuffle=True)
    initial_loader = DataLoader(initial_dataset, batch_size=batch_size, shuffle=True)
    pde_loader = DataLoader(pde_dataset, batch_size=batch_size, shuffle=True)

    num_batches = max(len(boundary_loader), len(initial_loader), len(pde_loader))

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_counter = 0
        for (xb_batch,), (xi_batch,), (xpde_batch,) in zip( \
                                                        itertools.cycle(boundary_loader), \
                                                        itertools.cycle(initial_loader),\
                                                        pde_loader):
            optimizer.zero_grad()
            loss = total_loss(model, xb_batch, xi_batch, xpde_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_counter += 1
            if batch_counter >= num_batches:
                break
        epoch_loss /= batch_counter
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
        loss_history.append(epoch_loss)
        epoch_history.append(epoch)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
            # Optionally save the best model
            torch.save(model.state_dict(), "best_model.pt")
            training_state = epoch
        else:
            counter += 1
        if counter > patience:
            print(f"Early stopping at epoch {epoch}")
            break
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
            if epoch > 1000:
                plot_loss_history(loss_history, epoch_history)


        if epoch % 100 == 0:
            # print(f"Epoch {epoch}, Loss: {loss.item()}")
            if epoch>100:
                plot_loss_history(loss_history, epoch_history)

    print(f"Boundary Loss: {loss_boundary(model, x_b).item()}")
    print(f"Initial Condition Loss: {loss_initial(model, x_i).item()}")
    print(f"PDE Loss: {loss_physics(model, x_pde).item()}")

    plot_loss_history(loss_history, epoch_history)

    plot_model_output(model, t=0)
    plot_model_output(model, t=0.1)

    # Save the model with name n_hidden and epochs
    model_name = f"pinn_model_{n_hidden}_{epochs}.pt"
    torch.save(model.state_dict(), model_name)
    print(f"Model taken from epoch {training_state} with best loss {best_loss:.4f}")
    print(f"Model saved as {model_name}")
    end = time.time()
    print(f"Training completed in {(end - start)/60:.2f} minutes")

