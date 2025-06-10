## Setup (for VSCode)

1. Make sure you have the `Dev Container` extension installed in VSCode and have `Docker` installed and running.

2. Open the command palette (Ctrl+Shift+P) and select `Dev Containers: Reopen in Container`. 

3. After the container is created, wait for the dependencies to be installed.

4. Run the `test_install.py` script to check if the installation is successful.

```bash
python test_install.py
```

## Complex mode

- If you want to run `Jupyter Notebooks`, set the kernel in the top right corner of the notebook to be `dolfinx-env (Python 3.12.3)` (real mode) or `Python 3 (DOLFINx complex)` (complex mode).

- If you want to run files in the terminal, use

```bash
source /usr/local/bin/dolfinx-complex-mode
source /usr/local/bin/dolfinx-real-mode
```

to switch between real and complex builds of DOLFINx/PETSc.

