# Start from the DOLFINx lab image which already includes Jupyter
FROM dolfinx/lab:stable

# (Re)install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with GPU support
RUN pip3 install torch torchvision torchaudio

# Install numpy and matplotlib
RUN pip3 install numpy matplotlib tqdm jupyter  ipympl

# # Duplicate the DOLFINx python install (experimental)
RUN cp -r /dolfinx-env/share/jupyter/kernels/python3 /dolfinx-env/share/jupyter/kernels/python3-complex

# # Add DOLFINx kernel specifications (experimental)
COPY complex-kernel.json /dolfinx-env/share/jupyter/kernels/python3-complex/kernel.json
# COPY complex-kernel.json /usr/local/share/jupyter/kernels/python3-complex/kernel.json

# Verify installations
RUN python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" && \
    python3 -c "import dolfinx; print('DOLFINx version:', dolfinx.__version__)"
