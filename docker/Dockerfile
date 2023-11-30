# This is current latest torchserve image, set specific version just in case of futre releases
FROM pytorch/torchserve:0.9.0-gpu

# Switch to root to install nvcc
USER root

# Install nvcc needed to compile custom VToonify cuda libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    cuda-nvcc-12-1 \
    libcusparse-dev-12-1 \
    libcublas-dev-12-1 \
    libcusolver-dev-12-1 \
    && rm -rf /var/lib/apt/lists/*

# Go back to original user to dont run as root
USER model-server
# maybe necessary to access nvcc
# RUN usermod -a -G video model-server

WORKDIR crypsis-delizziosa-model/

# Install python dependencies
RUN pip install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Adding config file needed for both running and compiling the model
ADD config.properties config.properties

# # Download the checkpoints
# ADD scripts/download_checkpoints.sh download_checkpoints.sh
# RUN ./download_checkpoints.sh
