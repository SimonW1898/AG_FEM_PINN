import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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
    u = model(x)
    grads = get_gradients(model, x)
    dx1 = grads[:, :, 0]
    dx2 = grads[:, :, 1]
    dt = grads[:, :, 2]

    u_r_x1 = dx1[:, 0]
    u_r_x2 = dx2[:, 0]
    u_r_x1x1 = second_derivative(u_r_x1, x, 0)
    u_r_x2x2 = second_derivative(u_r_x2, x, 1)
    lap_u_r = u_r_x1x1 + u_r_x2x2

    u_i_x1 = dx1[:, 1]
    u_i_x2 = dx2[:, 1]
    u_i_x1x1 = second_derivative(u_i_x1, x, 0)
    u_i_x2x2 = second_derivative(u_i_x2, x, 1)
    lap_u_i = u_i_x1x1 + u_i_x2x2

    u_r_t = dt[:, 0]
    u_i_t = dt[:, 1]

    res_real = -u_i_t + lap_u_r
    res_imag = u_r_t + lap_u_i

    loss = torch.mean(res_real**2 + res_imag**2)
    return loss

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

def total_loss(model, x_b, x_i, x_pde):
    lambda_bc = 1.0
    lambda_ic = 1.0
    lambda_pde = 1.0
    loss_bc = loss_boundary(model, x_b)
    loss_ic = loss_initial(model, x_i)
    loss_pde = loss_physics(model, x_pde)
    total_loss = lambda_bc * loss_bc + lambda_ic * loss_ic + lambda_pde * loss_pde
    return total_loss

# Sampling functions
T = 0.1

def sample_boundary_points(num_points, T=0.1):
    x1 = np.zeros(num_points)
    x2 = np.zeros(num_points)
    t = np.random.uniform(0.01, T, num_points)
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




# Training
model = PINN(n_hidden=128)
N_points = 15000
x_b = sample_boundary_points(N_points, T)
x_i = sample_initial_points(N_points)
x_pde = sample_interior_points(N_points, T)

optimizer = optim.Adam(
    params=model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0,
    amsgrad=False
)


#  Plotting function
def plot_model_output(model, t=0):
    x1 = np.linspace(0, 1, 100)
    x2 = np.linspace(0, 1, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.column_stack((X1.ravel(), X2.ravel(), np.full(X1.size, t)))
    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    u_pred = model(X_tensor).detach().numpy()
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

epochs = 10000
loss_history = []
epoch_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = total_loss(model, x_b, x_i, x_pde)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        loss_history.append(loss.item())
        epoch_history.append(epoch)
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print(f"Boundary Loss: {loss_boundary(model, x_b).item()}")
print(f"Initial Condition Loss: {loss_initial(model, x_i).item()}")
print(f"PDE Loss: {loss_physics(model, x_pde).item()}")

plt.plot(epoch_history, loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss History')
plt.grid()
plt.savefig("loss_history.png")
plt.close()

plot_model_output(model, t=0)
plot_model_output(model, t=0.1)