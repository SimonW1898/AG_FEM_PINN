import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import itertools

import numpy as np
import matplotlib.pyplot as plt
import optuna

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

def loss_initial(model, x_initial):
    u_initial = model(x_initial)
    initial_r = torch.sin(np.pi * x_initial[:, 0]) * torch.sin(np.pi * x_initial[:, 1])
    initial_i = torch.zeros_like(initial_r)
    res_r = u_initial[:, 0] - initial_r
    res_i = u_initial[:, 1] - initial_i
    loss = torch.mean(res_r**2 + res_i**2)
    return loss

def total_loss(model, x_b, x_i, x_pde, lambda_bc=1.0, lambda_ic=1.0, lambda_pde=1.0):
    loss_bc = loss_boundary(model, x_b)
    loss_ic = loss_initial(model, x_i)
    loss_pde = loss_physics(model, x_pde)

    sum_lambda = lambda_bc + lambda_ic + lambda_pde
    total_loss = (lambda_bc * loss_bc + lambda_ic * loss_ic + lambda_pde * loss_pde) / sum_lambda
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


def make_sample_points_file_name(n_points,T = 1):
    N_points_interior = int(n_points)
    N_points_initial = int(n_points/5)
    N_points_boundary = int(n_points/5)
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



def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    # n_hidden []
    n_hidden = trial.suggest_categorical('n_hidden', [64, 128, 256, 512])
    n_layers = trial.suggest_int('n_layers', 2, 5)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

    # Regularization parameters
    lambda_min = 1e-1
    lambda_max = 1.0

    lambda_bc = trial.suggest_categorical('lambda_bc', [lambda_min, lambda_max])
    lambda_ic = trial.suggest_categorical('lambda_ic', [lambda_min, lambda_max])
    lambda_pde = trial.suggest_categorical('lambda_pde', [lambda_min, lambda_max])

    # Create model
    activation = 'tanh'
    model = PINN(n_hidden=n_hidden, n_layers=n_layers, activation=activation)
    model.to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Define parameters
    n_epochs = 500
    n_points = 50000
    T = 1


    # load sample points
    x_b, x_i, x_pde = load_sample_points(f"sample_points_{n_points}.npz")   
    x_b = x_b.to(device).requires_grad_()
    x_i = x_i.to(device).requires_grad_()
    x_pde = x_pde.to(device).requires_grad_()

    # get validation points for pde loss
    n_val_points = 10000
    _,_, x_val_pde = load_sample_points(f"sample_points_{n_val_points}.npz")
    x_val_pde = x_val_pde.to(device).requires_grad_()

    # make DataLoaders
    batch_size = 1024  # You can adjust this
    boundary_dataset = TensorDataset(x_b)
    initial_dataset = TensorDataset(x_i)
    pde_dataset = TensorDataset(x_pde)
    # Create DataLoaders
    boundary_loader = DataLoader(boundary_dataset, batch_size=batch_size, shuffle=True)
    initial_loader = DataLoader(initial_dataset, batch_size=batch_size, shuffle=True)
    pde_loader = DataLoader(pde_dataset, batch_size=batch_size, shuffle=True)

    # handle different number of batches
    batch_counter = 0
    num_batches = max(len(boundary_loader), len(initial_loader), len(pde_loader))

    # initialize for convergence eval
    # loss_history = []
    # epoch_history = []
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        batch_counter = 0
        for (xb_batch,), (xi_batch,), (xpde_batch,) in zip(
            itertools.cycle(boundary_loader),
            itertools.cycle(initial_loader),
            pde_loader
        ):
            optimizer.zero_grad()
            loss = total_loss(model, xb_batch, xi_batch, xpde_batch, lambda_bc, lambda_ic, lambda_pde)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_counter += 1
            if batch_counter >= num_batches:
                break
        epoch_loss /= batch_counter

        # compute validation
        if epoch % 20 == 0:
            model.eval()
            val_loss = loss_physics(model, x_val_pde)
            trial.report(val_loss.item(), step = epoch)
            # check if trial should be pruned
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
    model.eval()
    val_loss = loss_physics(model, x_val_pde)

    return val_loss.item()


def main_tuning():    # Number of trials for hyperparameter optimization
    n_trials = 10

    # Create a study object
    study = optuna.create_study(direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    
    study.optimize(objective, n_trials=n_trials, timeout= None)

    print("Best trial:")
    trial = study.best_trial
    print(f"Validation Loss: {trial.value}")
    print("Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main_tuning()