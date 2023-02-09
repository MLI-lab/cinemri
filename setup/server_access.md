# Setup procedure personalized for Johannes

1. Copy `.bash_aliases` to the current working directory and `ssh_config` to the `.ssh` folder.
2. Connect to the server
```bash
ssh gpumlp2
```
3. Start a docker container
```bash
launch-docker
```
4. Install the necessary libraries
```bash
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
pip install fastmri
pip install h5py
pip install torchkbnufft



apt-get update
apt-get install --yes git nano
apt-get install --yes screen 
apt-get install --yes gcc make libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev gfortran

apt-get install --yes --reinstall build-essential
pip install image-quality
pip install pyrtools
pip install -U csaps
pip install torchviz
apt-get install --yes graphviz

cd /workspace
git clone https://github.com/mrirecon/bart.git
cd bart
# sed -i 's/CUDA=0/CUDA=1/' Makefile
make
export PATH=$PATH:/workspace/bart
export TOOLBOX_PATH=/workspace/bart

apt install ffmpeg
apt install zip unzip

# git clone https://github.com/mrirecon/bartpy.git
# pip install -r requirements.txt
# python3 setup.py install


```
5. Start the jupyter notebook
```bash
launch-jupyter
```
6. Detach from the container: CRTL+P, then CTRL+Q !do not press CTRL+D!
7. Re-enter container if necessary
```bash
enter-docker
```
8. Step 5. may be skipped of VSCode is connected to the Container directly using the Remote Container and Remote SSH extensions.
9. Check that the working directory of your notebooks is the base directory of this repository:
```python
# verify that the current working directory is the root of the Git repo.
# Otherwise, not all imports can be resolved.
# The cwd can be changed in the VSCode settings: Jupyter -> Notebook File Root (change ${fileDirname} to correct path)
print(os.getcwd())
```

Warning when creating the docker container:

WARNING: Your kernel does not support swap limit capabilities or the cgroup is not mounted. Memory limited without swap.