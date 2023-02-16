from types import SimpleNamespace
import numpy as np
import math
import torch
import PIL.Image
import os, sys
import h5py
import importlib

os.environ["TOOLBOX_PATH"] = "/workspace/bart"
sys.path.insert(0,os.environ["TOOLBOX_PATH"] + "/python/")
import bart

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



def rss(output): # shape of output: (Ns, Nc, Ny, Nx) or (Nc, Ny, Nx)
    """
    Compute root sum of squares (RSS) for two channel output.
    """
    abs = torch.square(output)
    return torch.sqrt(torch.sum(abs, dim=-3))


def image_cross_section(imgs, x=140, y_range=[50,350], angle=50):
    cross_section = np.zeros((len(imgs), y_range[1] - y_range[0]))
    for i, raw_img in enumerate(imgs):
        im = PIL.Image.fromarray(raw_img)
        im = im.rotate(angle)
        im_array = np.array(im)
        cross_section[i,:] = im_array[y_range[0]:y_range[1], x:x+1].T
    return cross_section

# variable density grid based on cava
def cava_grid(N, s, k=3.0):
    Ns = math.ceil(N / s)
    c = (N/2-Ns/2)/(math.pow(Ns/2, k))
    pc = np.zeros((Ns), dtype=np.int32)

    for i in range(1, Ns+1):
        if N%2==0: # if even, shift by 1/2 pixel
            indC = i - c*np.sign((Ns/2+1/2)-i)*math.pow(abs((Ns/2+1/2)-i),k) + (N-Ns)/2 + 1/2
            indC = indC - N*(indC>=(N+0.5))
        else:
            indC = i - c*np.sign((Ns/2+1/2)-i)*math.pow(abs((Ns/2+1/2)-i),k) + (N-Ns)/2
        pc[i-1] = int(round(indC)) - 1

    assert len(set(pc)) == len(pc)
    return pc, Ns

# implementation: https://github.com/OSU-CMR/GRO-CAVA/blob/master/cava_fun.m
def cava(N, s, D=1, k=3):
    # s = 2
    # k = 3 # alpha
    # N = 200 # grid size
    # D = 3 # number of dynamics


    gr = (math.sqrt(5) + 1 ) / 2 # golden ratio
    Ns = math.ceil(N / s)
    c = (N/2-Ns/2)/(math.pow(Ns/2, k))

    pe = np.zeros((Ns, D))
    pc = np.zeros((Ns, D), dtype=np.int32)

    for e in range(D):
        for i in range(Ns):
            if i==0:
                pe[i,e] = (math.floor(Ns/2) + 1 + e*math.sqrt(11)*Ns/(D*gr) - 1) % Ns + 1
            elif i > 0:
                pe[i,e] = ((pe[i-1,e] + Ns/gr)-1)%Ns + 1

            pe[i,e] = pe[i,e] - Ns * (pe[i,e] >= (Ns+0.5))

            if N%2==0: # if even, shift by 1/2 pixel
                indC = pe[i,e] - c*np.sign((Ns/2+1/2)-pe[i,e])*math.pow(abs((Ns/2+1/2)-pe[i,e]),k) + (N-Ns)/2 + 1/2
                indC = indC - N*(indC>=(N+0.5))
            else:
                indC = pe[i,e] - c*np.sign((Ns/2+1/2)-pe[i,e])*math.pow(abs((Ns/2+1/2)-pe[i,e]),k) + (N-Ns)/2

            pc[i,e] = int(round(indC)) - 1

    for e in range(D):
        if len(set(pc[:,e])) != pc.shape[0]:
            print("Warning: some lines are sampled multiple times per dynamic")

    return pc, Ns


def cava_generator(Nl, N, s, k=3., D=1, e=0):
    # Nl: number of lines generated
    # N: grid size
    # s: subsampling rate (acceleration factor)
    # k: variable density distribution parameter: higher -> center sampled more densely
    # D: total number of dynamics (not used in original paper)
    # e: current number of the dynamic  (not used in original paper)

    # maps (0. Ns]
    
    gr = (math.sqrt(5) + 1 ) / 2 # golden ratio
    Ns = math.ceil(N / s)
    
    for n in range(Nl):
        if n == 0:
            pe = (math.floor(Ns/2) + e*math.sqrt(11)*Ns/(D*gr)) % Ns
        else:
            pe = ((pe_last + Ns/gr))%Ns
        pe_last = pe

        if pe == 0.:
            yield 0

        yield cava_mapping(pe, N, Ns, k=k) - 1 # -1 is a modification to obtain values in the range from 0 to N-1. According to the paper, the input range 1...Ns should be mapped to 1...N, but this is not the case. Instead, [0...Ns) is mapped to [0, N]


def cava_mapping(pe, N, Ns, k=3.0):
    c = (N/2-Ns/2)/(math.pow(Ns/2, k))
    return np.ceil(pe - c*np.sign(Ns/2-pe)*np.power(np.abs(Ns/2-pe),k) + (N-Ns)/2)


def get_number_of_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def conv2d_out_size(in_size, kernel=3, stride=1, dilation=1, padding=0):
    h = math.floor((in_size[0] + 2 * padding - dilation * (kernel - 1) - 1)/stride + 1)
    w = math.floor((in_size[1] + 2 * padding - dilation * (kernel - 1) - 1)/stride + 1)
    return h, w


def create_dir(path):
    path = os.path.normpath(path)
    segments = path.split(os.sep)
    for i in range(1, len(segments)+1):
        if i==1 and segments[0] == "": continue
        if not os.path.isdir("/".join(segments[0:i])):
            os.mkdir("/".join(segments[0:i]))


def bart_transform(sample):
    (Nk, Nc, Nz, Ny, Nx) = sample["shape"]

    kspace = to_numpy(sample["kspace"])
    smaps = to_numpy(sample["smaps"])
    
    return {
        "kspace": kspace.T.reshape(Nx, Ny, Nz, Nc, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, Nk),
        "smaps": smaps.T.reshape(Nx, Ny, Nz, Nc, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, Nk)
    }

def bart_l1_wavelet_reconstruction(sample, regularization_parameter=0.0001):
    bart_sample = bart_transform(sample)
    return bart.bart(1, "pics -l1 -r{} -e -D -S".format(regularization_parameter), bart_sample["kspace"], bart_sample["smaps"]).T

def bart_tv_reconstruction(sample, regularization_parameter=0.0001):
    bart_sample = bart_transform(sample)
    return bart.bart(1, "pics -R T:3:0:{} -e -D -S".format(regularization_parameter), bart_sample["kspace"], bart_sample["smaps"]).T


def rss_reconstruction(kspace):
    """ 
    Performs a Root-sum-of-squares recontruction. The required kspace format is: np.complex (..., Nc, Nz, Ny, Nx).
    """
    assert kspace.ndim >= 4
    kspace = to_tensor(kspace)
    imgs = ifft2(kspace)
    img_rec = torch.sqrt(torch.sum(torch.sum(torch.square(imgs), dim=-5), dim=-1))
    return img_rec.numpy()

def flatten_param(param, flattened=dict(), prefix=""):
    for k, v in param.__dict__.items():
        if isinstance(v, SimpleNamespace):
            flatten_param(v, flattened=flattened, prefix=k)
        else:
            flattened[prefix + "." + str(k)] = v
    return flattened

def to_hparam_dict(param):
    flattened = flatten_param(param)
    filtered = dict()
    for k, v in flattened.items():
        if isinstance(v, int) or isinstance(v, float) or isinstance(v, str) or isinstance(v, bool) or torch.is_tensor(v):
            filtered[k] = v
        else:
            filtered[k] = "{}".format(v) # print to str
    return filtered

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
            copy[k] = entry[k].to("cuda") # does detach make it faster?
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