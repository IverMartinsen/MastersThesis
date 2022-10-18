# A dockerfile must always start by importing the base image.
# We use the keyword 'FROM' to do that.
# In our example, we want import the python image.
# So we write 'python' for the image name and 'latest' for the version.
# Note that we also specify a platform?
FROM --platform=amd64 conda/miniconda3:latest

# The WORKDIR command sets the working directory for subseqent command.
WORKDIR /src

COPY requirements.yml /src

# The RUN command specifyes what will be run when the image is created.
# In this case we install mamba and create a python environment.
RUN conda install mamba -n base -c conda-forge
RUN mamba env create --file requirements.yml