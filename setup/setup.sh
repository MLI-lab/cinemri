conda install --yes jupyter
conda install --yes PyWavelets
conda install --yes skimage
conda install --yes scikit-image
conda install --yes matplotlib
conda install --yes -c conda-forge pytorch-lightning
conda install --yes scikit-image=0.16
conda install --yes pandas
conda install --yes pydoc-markdown
conda install --yes -c conda-forge image-quality
conda install --yes sympy
pip install fastmri
pip install h5py
pip install tikzplotlib
pip install docstr_md
apt-get update
apt-get install git nano
apt-get install gcc make libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev gfortran

apt-get install --yes --reinstall build-essential
pip install image-quality
pip install tensorflow
pip install torchkbnufft
pip install csaps
pip install pydicom
pip install moviepy
conda install --yes -c conda-forge tensorboard=2.9.1

# install bart toolbox for MRI reconstruction
cd /workspace
git clone https://github.com/mrirecon/bart.git
cd bart
make
export PATH=$PATH:/workspace/bart
export TOOLBOX_PATH=/workspace/bart

apt install ffmpeg
apt install zip unzip
pip install pyrtools

# install tool for encoding mp4 videos with a variable frame rate
cd /workspace
git clone https://github.com/nu774/mp4fpsmod.git
cd mp4fpsmod
apt install --yes autoconf
apt install --yes libtool
./bootstrap.sh
./configure
make
make install

# installation of VUE and gitlab-runner for the deployment of the supplementary website
cd /workspace
apt get update
apt install --yes curl
apt install --yes npm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash
source ~/.bashrc
nvm install v16.19.0
npm install -g @vue/cli
curl -LJO "https://gitlab-runner-downloads.s3.amazonaws.com/latest/deb/gitlab-runner_amd64.deb"
dpkg -i gitlab-runner_amd64.deb
