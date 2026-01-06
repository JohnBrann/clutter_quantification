# Dockerfile for grasp-data-collection
FROM ros:noetic-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# System deps (build tools + GL/X11 libs for PyBullet/OpenCV/Open3D)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git cmake build-essential pkg-config python3-dev \
    libeigen3-dev libpcl-dev libyaml-cpp-dev libgoogle-glog-dev \
    libhdf5-dev libgflags-dev openmpi-bin libopenmpi-dev \
    libgl1 libglib2.0-0 libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
 && rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Accept Anaconda channels ToS (non-interactive)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create env and make it the default
RUN conda create -y -n grasp-data-collection python=3.8 && conda clean -afy
ENV CONDA_DEFAULT_ENV=grasp-data-collection
ENV PATH=/opt/conda/envs/grasp-data-collection/bin:$PATH

# (Optional) copy reqs if you want; we’ll still pin our known-good stack after
COPY requirements.txt /tmp/requirements.txt

# Core Python deps (pin versions that worked for you)
RUN pip install --no-cache-dir -r /tmp/requirements.txt || true \
 && pip install --no-cache-dir \
    numpy==1.23.5 scipy==1.10.1 tqdm==4.66.4 \
    open3d==0.17.0 pybullet==3.2.5 opencv-python==4.7.0.72 \
    h5py==3.10.0 pyyaml==6.0.1 matplotlib==3.7.2 scikit-learn==0.24.2

# Torch 2.2.2 + CUDA 11.8
RUN pip install --no-cache-dir \
    torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric 2.6.1 + matching extension wheels for torch 2.2 cu118
RUN pip install --no-cache-dir torch_geometric==2.6.1 \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

# Make ROS + conda env available in non-interactive shells too
SHELL ["/bin/bash", "-lc"]
RUN echo 'source /opt/ros/noetic/setup.bash' >> /etc/bash.bashrc \
 && echo 'source /opt/conda/etc/profile.d/conda.sh && conda activate grasp-data-collection' >> /etc/bash.bashrc

# Workdir (mounted at runtime)
WORKDIR /grasp-data-collection

# Handy envs for X11 (optional but often helps)
ENV QT_X11_NO_MITSHM=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics,display

# ensure we’re installing into the conda env
ENV PATH=/opt/conda/envs/grasp-data-collection/bin:$PATH

# build deps for convonet_setup.py
RUN python -m pip install --no-cache-dir "Cython>=0.29" pybind11 cmake ninja "numpy<2.0"
RUN pip install --no-cache-dir "scikit-image==0.20.0"
RUN pip install --no-cache-dir trimesh urchin

# Default shell
CMD ["bash"]
