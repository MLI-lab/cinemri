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
chmod +x ./setup/setup.sh
./setup/setup.sh
```
5. Check that the working directory of your notebooks is the base directory of this repository:
```python
# verify that the current working directory is the root of the Git repo.
# Otherwise, not all imports can be resolved.
# The cwd can be changed in the VSCode settings: Jupyter -> Notebook File Root (change ${fileDirname} to correct path)
print(os.getcwd())
```

## Setup using Dockerfile
The current dockerfile is outdated. It needs to be updated by adding the changes that were made in setup.sh.