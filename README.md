## Setup (for VSCode)

1. Make sure you have the `Dev Container` extension installed in VSCode and have `Docker` installed and running.

2. Open the command palette (Ctrl+Shift+P) and select `Dev Containers: Reopen in Container`. 

3. After the container is created, wait for the dependencies to be installed.

4. Run the `test_installation.py` script to check if the installation is successful.

```bash
python test_installation.py
```

## Complex mode

- If you want to run `Jupyter Notebooks`, set the kernel in the top right corner of the notebook to be `dolfinx-env (Python 3.12.3)` (real mode) or `Python 3 (DOLFINx complex)` (complex mode).

- If you want to run files in the terminal, use

```bash
source /usr/local/bin/dolfinx-complex-mode
source /usr/local/bin/dolfinx-real-mode
```

to switch between real and complex builds of DOLFINx/PETSc.

## Known issues

- After building and opening the container, the `Jupyter` kernels might not be available.

## Workarounds and temporary fixes

**Access remote Jupyter server in your local workspace**: Run the pre-built Docker container that has the kernels installed for `JupyterLab` locally.

```bash
docker run --init -ti -p 8888:8888 dolfinx/lab:stable
```

It will print a `JupyterLab` URL with a token in the logs (e.g., `http://127.0.0.1:8888/lab?token=...`). In your local workspace, open the `Jupyter Notebook` interface, click on `Select Kernel` -> `Existing Jupyter Server`, and paste the URL and token. You can then select the Dolfinx kernels from the list.
