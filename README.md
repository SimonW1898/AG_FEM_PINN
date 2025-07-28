# AG_FEM_PINN: On the Numerical Treatment of the Time-Dependent Schrödinger Equation in Two Dimensions

This repository contains implementations of both Finite Element Method (FEM) and Physics-Informed Neural Networks (PINNs) for solving the time-dependent Schrödinger equation. The project provides a comprehensive comparison between traditional numerical methods and modern machine learning approaches for quantum mechanical simulations.

## Overview

The project solves the time-dependent Schrödinger equation:

$$i \frac{\partial u}{\partial t} = -\Delta u + V(x,t)u$$

with homogeneous Dirichlet boundary conditions on the unit square [0,1]², where:
- $u(x,t)$ is the complex wavefunction
- $V(x,t)$ is the potential energy function
- $\Delta$ is the Laplacian operator

## Main Components

### 1. Finite Element Method (FEM) Implementation (`fem/`)

The FEM implementation provides a robust, high-accuracy numerical solution using DOLFINx.

#### Key Features:
- **Modular Design**: Class-based architecture with `SchrodingerSolver` and `StationarySchrodingerSolver`
- **Multiple Time Schemes**: Backward Euler time discretization
- **Flexible Potentials**: Support for various potential functions including:
  - Harmonic oscillator potential
  - Double well potential
  - Model potentials with customizable parameters
  - Time-dependent laser pulse interactions
- **Eigenvalue Analysis**: Stationary state solver for ground state and excited states
- **Visualization**: 2D/3D plotting, animations, and error analysis
- **Data Export**: Save solutions in NPZ format for post-processing

#### Main Classes:
- `SchrodingerSolver`: Time-dependent Schrödinger equation solver
- `StationarySchrodingerSolver`: Eigenvalue problem solver for stationary states
- `ModelPotential`: Configurable potential energy functions
- `LaserPulse`: Time-dependent laser-dipole coupling

### 2. Physics-Informed Neural Networks (PINN) Implementation (`pinns/`)

The PINN implementation uses PyTorch to solve the Schrödinger equation using neural networks that are trained to satisfy the underlying physics.

#### Key Features:
- **Neural Network Architecture**: Fully connected networks with configurable layers and activation functions
- **Physics-Informed Loss**: Loss function incorporating the Schrödinger equation residual
- **Multi-Objective Training**: Balanced loss terms for PDE, boundary conditions, and initial conditions
- **Hyperparameter Optimization**: Optuna-based automatic tuning
- **Flexible Sampling**: Configurable point sampling strategies for training
- **Visualization**: Real-time plotting and solution analysis

#### Main Functions:
- `PINN`: Neural network model class
- `loss_physics`: Physics-informed loss function
- `train_model`: Training function with configurable parameters
- `objective`: Optuna objective function for hyperparameter tuning

## Setup Instructions

### Prerequisites
- Docker installed and running
- VS Code with Dev Container extension

### Installation

1. **Open in Dev Container**:
   - Open the command palette (`Ctrl+Shift+P` or `Cmd+Shift+P`)
   - Select `Dev Containers: Reopen in Container`
   - Wait for the container to build and dependencies to install

2. **Test Installation**:
   ```bash
   python test_installation.py
   ```

3. **Jupyter Notebook Setup**:
   - For real mode: Set kernel to `dolfinx-env (Python 3.12.3)`
   - For complex mode: Set kernel to `Python 3 (DOLFINx complex)`

4. **Terminal Mode Switching**:
   ```bash
   # Switch to complex mode
   source /usr/local/bin/dolfinx-complex-mode
   
   # Switch to real mode  
   source /usr/local/bin/dolfinx-real-mode
   ```

### Alternative: Local Jupyter Server
If kernels are not available in the container:
```bash
docker run --init -ti -p 8888:8888 dolfinx/lab:stable
```
Then connect to the JupyterLab URL with token from the logs.

## File Structure

```
AG_FEM_PINN/
├── fem/                    # Finite Element Method implementation
│   ├── main_fem.py        # Main FEM solver classes
│   ├── potentials.py      # Potential energy functions
│   ├── analysis.py        # Analysis and plotting utilities
│   └── figures/           # Generated FEM figures
├── pinns/                 # Physics-Informed Neural Networks
│   ├── main_pinn.py       # Main PINN implementation
│   ├── pinn_plot.py       # PINN visualization utilities
│   └── figures/           # Generated PINN figures
├── report/                # LaTeX report and documentation
├── test_installation.py   # Installation verification script
└── README.md             # This file
```

## Dependencies

- **DOLFINx**: Finite element library
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **Matplotlib**: Plotting and visualization
- **Optuna**: Hyperparameter optimization
- **SLEPc**: Eigenvalue solvers
- **PETSc**: Linear algebra backend

## Known Issues

- Jupyter kernels may not be available immediately after container build
- Complex mode requires specific kernel selection
- GPU acceleration for PINNs requires CUDA-compatible PyTorch installation

## Contributing

This project is designed for research and educational purposes. Contributions are welcome, particularly for:
- Additional potential functions
- New time integration schemes
- Improved PINN architectures
- Enhanced visualization capabilities
- Performance optimizations

## License

This project is for academic research purposes. Please cite appropriately if used in publications.

## Authors
Lasse Kreimendahl and Simon Wenchel
