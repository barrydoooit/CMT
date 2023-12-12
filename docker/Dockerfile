FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y

## Install miniconda
# ENV CONDA_DIR /opt/conda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#      /bin/bash ~/miniconda.sh -b -p /opt/conda
# # Put conda in path so we can use conda activate
# ENV PATH=$CONDA_DIR/bin:$PATH
# RUN conda install -y python=3.8
# RUN conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch

RUN apt-get update && apt-get install -y \
    git nano build-essential cmake python3-dev  python3-pip python3-pip python3-tk ninja-build &&\
    rm -rf /var/lib/apt/lists/*

RUN ln -sv /usr/bin/python3 /usr/bin/python
ENV PATH="/root/.local/bin:${PATH}"
RUN wget -O /root/get-pip.py https://bootstrap.pypa.io/get-pip.py && python3 /root/get-pip.py --user

# PyTorch for CUDA 11.3
RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 --extra-index-url https://download.pytorch.org/whl/cu113
ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX;Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing;8.6"

# Python dependencies
RUN pip install Pillow==8.4.0
RUN pip install tqdm
RUN pip install torchpack
RUN pip install mmcv==1.6.0 mmcv-full==1.6.0 mmdet==2.24.0 mmsegmentation==0.29.1
RUN pip install mmdet3d==1.0.0rc5 spconv-cu113
RUN pip install flash-attn==0.2.2

RUN pip install nuscenes-devkit
RUN pip install mpi4py==3.0.3

# required by mmdet3d==1.0.0rc5 
RUN pip install numba==0.53.0

RUN pip install numpy==1.23.0
# RUN pip install ninja

RUN git clone https://github.com/barrydoooit/CMT.git
WORKDIR CMT

# RUN test -f /usr/local/cuda/include/cuda_runtime_api.h || (echo "cuda_runtime_api.h not found" && false)

ENV CPATH=/usr/local/cuda/include:$CPATH
ENV LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda

# RUN nvcc --version || (echo "nvcc not found" && false)
# RUN which nvcc
# RUN echo '#include <cuda_runtime_api.h>\nint main() {return 0;}' > test.cu && nvcc test.cu

# RUN python setup.py develop

# USER root
# RUN apt-get update && apt-get install -y openssh-server
# RUN mkdir /var/run/sshd
# COPY /home/barryding/.ssh/others/dockers.pub /root/.ssh/authorized_keys
# RUN chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
# RUN echo 'root:docker123' | chpasswd
# RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# EXPOSE 22
# CMD ["/usr/sbin/sshd", "-D"]