import torch
import os, sys
import h5py
import importlib

from .transforms import *

dtype = torch.cuda.FloatTensor

def pca(data, ndim=2, return_v=False):
    """
    Compute `ndim` principle components of `data`. The shape of `data` is: (num_samples, num_features).
    """
    # data shape (num_samples, num_features)
    x = data - np.mean(data, axis=0)
    (u, s, v) = np.linalg.svd(x, compute_uv=True)

    embedded_coords = s[0:ndim] * u[:,0:ndim]

    # useful for plotting variance of latent variables: Var[z] = V^T \Sigma V, z~N(\mu, \Sigma)
    if return_v:
        return embedded_coords, v[:, 0:ndim]
    
    return embedded_coords


# helper method to parse Matlab structs
def parse_struct(f):
    if isinstance(f, h5py._hl.group.Group):
        d = dict()
        for k in f.keys():
            if k == "AutoListeners__":
                continue
            d[k] = parse_struct(f[k])
        return d
    elif isinstance(f, h5py._hl.dataset.Dataset):
        return np.array(f).squeeze()
    return (type(f), f)

 # helper methods
def h5py2Complex(h5pyData, load_in_chunks=False):
    """
    Helper method for converting complex h5py arrays to numpy arrays.
    """
    if load_in_chunks: # good for coping with large kspace data
        shape = h5pyData.shape
        np_array = np.zeros(shape, dtype=np.csingle)
        for i in range(shape[0]):
            print('Loading chunk {} of {}'.format(i+1, shape[0]), '\r', end='')
            realImagTuples = np.array(h5pyData[i])
            np_array[i] = realImagTuples['real'] + 1j*realImagTuples['imag']
        return np_array

    else:
        realImagTuples = np.array(h5pyData)
        return np.array(realImagTuples['real'] + 1j*realImagTuples['imag'], dtype=np.csingle)


def copySampleToGPU(entry):
    copy = {}
    for k in entry.keys():
        if torch.is_tensor(entry[k]):
            copy[k] = entry[k].clone().to("cuda") # does detach make it faster?
        else:
            copy[k] = entry[k]
    return copy

def import_file(python_file_path):
    # import the model file
    spec = importlib.util.spec_from_file_location("module.name", os.path.abspath(python_file_path))
    file = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = file
    spec.loader.exec_module(file)
    return file

def create_dir(path):
    path = os.path.normpath(path)
    segments = path.split(os.sep)
    for i in range(1, len(segments)+1):
        if i==1 and segments[0] == "": continue
        if not os.path.isdir("/".join(segments[0:i])):
            os.mkdir("/".join(segments[0:i]))