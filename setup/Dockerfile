# Use an official Python runtime as a parent image
FROM pytorch/pytorch

# Set the working directory to /app
WORKDIR /workspace

RUN conda install --yes jupyter
RUN conda install --yes PyWavelets
RUN conda install --yes scikit-image
RUN conda install --yes matplotlib
RUN conda install --yes -c conda-forge pytorch-lightning
RUN conda install --yes scikit-image=0.16
RUN conda install --yes pandas
RUN conda install --yes -c conda-forge image-quality
RUN conda install --yes sympy
RUN pip install ffmpegio
RUN pip install fastmri
RUN pip install h5py
RUN pip install tikzplotlib
RUN pip install docstr_md
RUN apt-get update
RUN apt-get install --yes git nano
RUN apt-get install --yes gcc make libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev gfortran

RUN apt install --yes --reinstall build-essential
RUN pip install image-quality
RUN pip install tensorflow
RUN pip install torchkbnufft
RUN pip install csaps
RUN pip install pydicom
RUN conda install --yes tensorboard
RUN apt install --yes ffmpeg
RUN apt install --yes zip unzip
RUN pip install pyrtools
RUN pip install moviepy

RUN cd /workspace; git clone https://github.com/mrirecon/bart.git; cd bart; make; export PATH=$PATH:/workspace/bart
RUN export TOOLBOX_PATH=/workspace/bart


# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD /bin/bash
